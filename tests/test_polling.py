from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.db import connect, migrate
from rubik_spotify_queue.eventizer import Eventizer, next_capture_poll_seconds, next_poll_seconds
from rubik_spotify_queue.service import QueueService
from rubik_spotify_queue.spotify import PlaybackSnapshot
from rubik_spotify_queue.time_features import within_hour_window


def _settings() -> Settings:
    return Settings(
        ACTIVE_POLL_SECONDS=8,
        NEAR_TRACK_END_POLL_SECONDS=3,
        IDLE_POLL_SECONDS=120,
        QUIET_HOURS_POLL_SECONDS=900,
    )


def _snap(**kwargs: object) -> PlaybackSnapshot:
    data = {
        "observed_at": datetime.now(timezone.utc),
        "track_id": "abc",
        "track_name": "Song",
        "artist_id": "artist",
        "artist_name": "Artist",
        "album_id": None,
        "album_name": None,
        "duration_ms": 180_000,
        "progress_ms": 30_000,
        "is_playing": True,
        "device_id": None,
        "device_type": "phone",
        "shuffle_state": False,
        "context_uri": None,
        "spotify_uri": "spotify:track:abc",
        "raw_json": "{}",
    }
    data.update(kwargs)
    return PlaybackSnapshot(**data)


def test_quiet_window_uses_long_sleep_when_not_playing() -> None:
    assert next_poll_seconds(_settings(), None, queue_window_open=False) == 900
    assert next_poll_seconds(_settings(), _snap(is_playing=False), queue_window_open=False) == 900


def test_quiet_window_active_playback_uses_capture_sleep() -> None:
    assert next_poll_seconds(_settings(), _snap(), queue_window_open=False) == 8


def test_capture_poll_ignores_queue_window_sleep() -> None:
    assert next_capture_poll_seconds(_settings(), _snap()) == 8


def test_runtime_time_features_are_continuous() -> None:
    from rubik_spotify_queue.time_features import local_time_parts

    first = local_time_parts(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc), "UTC")
    second = local_time_parts(datetime(2026, 1, 5, 10, 45, tzinfo=timezone.utc), "UTC")
    assert first["hour_float"] == 10.0
    assert second["hour_float"] == 10.75
    assert first["hour_sin"] != second["hour_sin"]


def test_idle_uses_idle_sleep() -> None:
    assert next_poll_seconds(_settings(), None, queue_window_open=True) == 120
    assert next_poll_seconds(_settings(), _snap(is_playing=False), queue_window_open=True) == 120


def test_near_track_end_uses_fast_sleep() -> None:
    snap = _snap(progress_ms=170_000, duration_ms=180_000)
    assert next_poll_seconds(_settings(), snap, queue_window_open=True) == 3


def test_normal_playback_uses_active_sleep() -> None:
    assert next_poll_seconds(_settings(), _snap(), queue_window_open=True) == 8


def test_poll_result_log_is_readable(capsys) -> None:
    svc = QueueService.poller(_settings())
    svc.state.poll_count = 7
    svc._log_poll_result(_snap(track_name="Song", artist_name="Artist", progress_ms=12_000), 8)
    out = capsys.readouterr().out
    assert "poll #7" in out
    assert "Song" in out
    assert "Artist" in out
    assert "sleeping 8s" in out


def test_hour_window_supports_overnight_ranges() -> None:
    assert within_hour_window(23, 22, 3)
    assert within_hour_window(2, 22, 3)
    assert not within_hour_window(12, 22, 3)


def test_eventizer_records_position_in_session_and_resets_after_gap(tmp_path) -> None:
    db_path = tmp_path / "events.duckdb"
    migrate(db_path)
    con = connect(db_path)
    eventizer = Eventizer(_settings())
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

    try:
        eventizer.observe(con, _snap(track_id="A", track_name="A", observed_at=base))
        eventizer.observe(con, _snap(track_id="B", track_name="B", observed_at=base + timedelta(seconds=10)))
        eventizer.observe(con, _snap(track_id="C", track_name="C", observed_at=base + timedelta(minutes=31)))
        eventizer.observe(con, _snap(track_id="D", track_name="D", observed_at=base + timedelta(minutes=31, seconds=10)))

        rows = con.execute(
            """
            SELECT track_id, position_in_session, previous_track_id, session_id
            FROM listening_events
            ORDER BY started_at;
            """
        ).fetchall()
    finally:
        con.close()

    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("A", 1, None),
        ("B", 2, "A"),
        ("C", 1, None),
    ]
    assert rows[0][3] == rows[1][3]
    assert rows[1][3] != rows[2][3]

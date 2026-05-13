from __future__ import annotations

from datetime import datetime, timezone

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.eventizer import next_poll_seconds
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


def test_quiet_window_uses_long_sleep() -> None:
    assert next_poll_seconds(_settings(), _snap(), queue_window_open=False) == 900


def test_idle_uses_idle_sleep() -> None:
    assert next_poll_seconds(_settings(), None, queue_window_open=True) == 120
    assert next_poll_seconds(_settings(), _snap(is_playing=False), queue_window_open=True) == 120


def test_near_track_end_uses_fast_sleep() -> None:
    snap = _snap(progress_ms=170_000, duration_ms=180_000)
    assert next_poll_seconds(_settings(), snap, queue_window_open=True) == 3


def test_normal_playback_uses_active_sleep() -> None:
    assert next_poll_seconds(_settings(), _snap(), queue_window_open=True) == 8


def test_hour_window_supports_overnight_ranges() -> None:
    assert within_hour_window(23, 22, 3)
    assert within_hour_window(2, 22, 3)
    assert not within_hour_window(12, 22, 3)


from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pandas as pd

from rubik_spotify_queue.db import migrate
from rubik_spotify_queue.history_ingest import MODEL_COLUMNS, ingest_history
from rubik_spotify_queue.time_features import continuous_time_parts


def test_continuous_time_parts_include_minutes_and_day_fraction() -> None:
    base = datetime.fromisoformat("2026-01-05T10:00:00+00:00")
    later = datetime.fromisoformat("2026-01-05T10:30:00+00:00")

    first = continuous_time_parts(base, "UTC")
    second = continuous_time_parts(later, "UTC")

    assert first["hour_float"] == 10.0
    assert second["hour_float"] == 10.5
    assert second["day_float"] > first["day_float"]
    assert second["hour_sin"] != first["hour_sin"]
    assert second["day_sin"] != first["day_sin"]


def test_ingest_history_writes_model_and_debug_outputs(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {
            "ts": "2026-01-05T10:00:00Z",
            "platform": "linux",
            "ms_played": 180000,
            "spotify_track_uri": "spotify:track:one",
            "master_metadata_track_name": "One",
            "master_metadata_album_artist_name": "Artist",
            "master_metadata_album_album_name": "Album",
            "reason_start": "trackdone",
            "reason_end": "trackdone",
            "skipped": False,
            "shuffle": False,
            "offline": False,
            "incognito_mode": False,
        },
        {
            "ts": "2026-01-05T10:30:00Z",
            "platform": "linux",
            "ms_played": 60000,
            "spotify_track_uri": "spotify:track:two",
            "master_metadata_track_name": "Two",
            "master_metadata_album_artist_name": "Artist",
            "master_metadata_album_album_name": "Album",
            "reason_start": "fwdbtn",
            "reason_end": "fwdbtn",
            "skipped": True,
            "shuffle": False,
            "offline": False,
            "incognito_mode": False,
        },
    ]
    (raw / "Streaming_History_Audio_Test.json").write_text(json.dumps(rows), encoding="utf-8")

    result = ingest_history(
        raw_history_dir=raw,
        model_ready_dir=tmp_path / "model_ready",
        processed_dir=tmp_path / "processed",
        timezone_name="UTC",
    )

    full = pd.read_csv(result.model_ready_dir / "model_ready_history_full.csv")
    debug = pd.read_csv(result.processed_dir / "history_events_processed_debug.csv")

    assert list(full.columns) == MODEL_COLUMNS
    assert full["album_name"].tolist() == ["Album", "Album"]
    assert {"hour_float", "day_float"}.issubset(debug.columns)
    assert debug["hour_float"].tolist() == [10.0, 10.5]
    assert full["hour_sin"].nunique() == 2


def test_ingest_history_appends_live_duckdb_events(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {
            "ts": "2026-01-05T10:00:00Z",
            "platform": "linux",
            "ms_played": 180000,
            "spotify_track_uri": "spotify:track:one",
            "master_metadata_track_name": "One",
            "master_metadata_album_artist_name": "Artist",
            "master_metadata_album_album_name": "Album",
            "reason_end": "trackdone",
            "skipped": False,
            "incognito_mode": False,
        },
    ]
    (raw / "Streaming_History_Audio_Test.json").write_text(json.dumps(rows), encoding="utf-8")

    db_path = tmp_path / "live.duckdb"
    migrate(db_path)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            INSERT INTO songs (
                track_id, track_name, artist_id, artist_name, album_name, spotify_uri,
                first_seen_at, last_seen_at
            ) VALUES (
                'live-id', 'Live Track', 'artist-id', 'Live Artist', 'Live Album',
                'spotify:track:live-id', now(), now()
            );
            """
        )
        con.execute(
            """
            INSERT INTO listening_events (
                event_id, session_id, track_id, track_name, artist_id, artist_name,
                album_name, started_at, ended_at, max_progress_ms, wall_clock_ms,
                target_score, was_skipped, platform, shuffle_state, created_at
            ) VALUES (
                'event-1', 'session-1', 'live-id', 'Live Track', 'artist-id', 'Live Artist',
                'Live Album', TIMESTAMP '2026-01-06 10:00:00', TIMESTAMP '2026-01-06 10:03:00',
                180000, 180000, 0.9, false, 'ios', true, now()
            );
            """
        )
    finally:
        con.close()

    result = ingest_history(
        raw_history_dir=raw,
        model_ready_dir=tmp_path / "model_ready",
        processed_dir=tmp_path / "processed",
        timezone_name="UTC",
        live_events_db_path=db_path,
    )

    full = pd.read_csv(result.model_ready_dir / "model_ready_history_full.csv")
    debug = pd.read_csv(result.processed_dir / "history_events_processed_debug.csv")

    assert result.rows_full == 2
    assert "Live Album" in full["album_name"].tolist()
    assert "spotify:track:live-id" in debug["track_id"].tolist()

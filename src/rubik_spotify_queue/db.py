"""DuckDB schema and persistence helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb


DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id TEXT PRIMARY KEY,
        observed_at TIMESTAMP,
        track_id TEXT,
        track_name TEXT,
        artist_id TEXT,
        artist_name TEXT,
        album_id TEXT,
        album_name TEXT,
        duration_ms INTEGER,
        progress_ms INTEGER,
        is_playing BOOLEAN,
        device_id TEXT,
        device_type TEXT,
        shuffle_state BOOLEAN,
        context_uri TEXT,
        raw_json TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS listening_events (
        event_id TEXT PRIMARY KEY,
        session_id TEXT,
        track_id TEXT,
        track_name TEXT,
        artist_id TEXT,
        artist_name TEXT,
        album_id TEXT,
        album_name TEXT,
        started_at TIMESTAMP,
        ended_at TIMESTAMP,
        duration_ms INTEGER,
        max_progress_ms INTEGER,
        wall_clock_ms INTEGER,
        completion_ratio DOUBLE,
        target_score DOUBLE,
        was_skipped BOOLEAN,
        hour INTEGER,
        day_of_week INTEGER,
        hour_sin DOUBLE,
        hour_cos DOUBLE,
        day_sin DOUBLE,
        day_cos DOUBLE,
        platform TEXT,
        shuffle_state BOOLEAN,
        location_bucket TEXT,
        activity_bucket TEXT,
        previous_track_id TEXT,
        next_track_id TEXT,
        created_at TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS songs (
        track_id TEXT PRIMARY KEY,
        track_name TEXT,
        artist_id TEXT,
        artist_name TEXT,
        album_id TEXT,
        album_name TEXT,
        duration_ms INTEGER,
        spotify_uri TEXT,
        history_play_count INTEGER,
        history_avg_target_score DOUBLE,
        first_seen_at TIMESTAMP,
        last_seen_at TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS queue_actions (
        queue_action_id TEXT PRIMARY KEY,
        created_at TIMESTAMP,
        track_id TEXT,
        spotify_uri TEXT,
        predicted_score DOUBLE,
        model_version TEXT,
        status TEXT,
        detail TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_started_at ON listening_events(started_at);",
    "CREATE INDEX IF NOT EXISTS idx_events_track_id ON listening_events(track_id);",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_observed_at ON snapshots(observed_at);",
)

SCHEMA_UPGRADES: tuple[str, ...] = (
    "ALTER TABLE songs ADD COLUMN IF NOT EXISTS history_play_count INTEGER;",
    "ALTER TABLE songs ADD COLUMN IF NOT EXISTS history_avg_target_score DOUBLE;",
)


def connect(database_path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if not read_only:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database_path), read_only=read_only)


def migrate(database_path: Path) -> None:
    con = connect(database_path)
    try:
        for statement in DDL:
            con.execute(statement)
        for statement in SCHEMA_UPGRADES:
            con.execute(statement)
        con.commit()
    finally:
        con.close()

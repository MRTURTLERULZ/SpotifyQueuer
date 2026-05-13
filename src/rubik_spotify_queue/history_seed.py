"""Seed the service candidate universe from model-ready Spotify history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class SeedResult:
    rows_read: int
    songs_upserted: int
    queueable_songs: int


def seed_songs_from_model_ready(con: duckdb.DuckDBPyConnection, model_ready_dir: Path) -> SeedResult:
    full_path = model_ready_dir / "model_ready_history_full.csv"
    if not full_path.is_file():
        raise FileNotFoundError(full_path)

    frame = pd.read_csv(full_path)
    required = {"track_id", "artist_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{full_path} is missing required columns: {missing}")

    uri_lookup = _load_spotify_uri_lookup(model_ready_dir)
    songs = frame[["track_id", "artist_id"]].dropna().drop_duplicates(subset=["track_id", "artist_id"]).copy()
    now = datetime.now(timezone.utc)
    count = 0
    queueable = 0
    for row in songs.itertuples(index=False):
        track_id = str(row.track_id)
        artist_id = str(row.artist_id)
        spotify_uri = track_id if track_id.startswith("spotify:track:") else uri_lookup.get((track_id, artist_id))
        if spotify_uri:
            queueable += 1
        con.execute(
            """
            INSERT INTO songs (
                track_id, track_name, artist_id, artist_name, album_id, album_name,
                duration_ms, spotify_uri, first_seen_at, last_seen_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (track_id) DO UPDATE SET
                artist_id = COALESCE(EXCLUDED.artist_id, songs.artist_id),
                artist_name = COALESCE(EXCLUDED.artist_name, songs.artist_name),
                spotify_uri = COALESCE(EXCLUDED.spotify_uri, songs.spotify_uri),
                last_seen_at = EXCLUDED.last_seen_at;
            """,
            [
                track_id,
                None,
                artist_id,
                artist_id,
                None,
                None,
                None,
                spotify_uri,
                now,
                now,
            ],
        )
        count += 1

    return SeedResult(rows_read=len(frame), songs_upserted=count, queueable_songs=queueable)


def _load_spotify_uri_lookup(model_ready_dir: Path) -> dict[tuple[str, str], str]:
    debug_path = model_ready_dir.parent / "processed" / "history_events_processed_debug.csv"
    if not debug_path.is_file():
        return {}

    debug = pd.read_csv(debug_path)
    required = {"track_id", "track_name", "artist_id"}
    if not required.issubset(debug.columns):
        return {}

    out: dict[tuple[str, str], str] = {}
    for row in debug[["track_id", "track_name", "artist_id"]].dropna().itertuples(index=False):
        uri = str(row.track_id)
        if not uri.startswith("spotify:track:"):
            continue
        key = (str(row.track_name), str(row.artist_id))
        out[key] = uri
    return out

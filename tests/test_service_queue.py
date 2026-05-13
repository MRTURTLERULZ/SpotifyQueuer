from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.db import migrate
from rubik_spotify_queue.service import QueueService


class FakeSpotifyClient:
    def __init__(self) -> None:
        self.queued: list[str] = []

    def add_to_queue(self, spotify_uri: str) -> int:
        self.queued.append(spotify_uri)
        return 204


def test_queue_batch_adds_two_tracks_and_records_actions(tmp_path) -> None:
    db_path = tmp_path / "service.duckdb"
    migrate(db_path)
    con = duckdb.connect(str(db_path))
    now = datetime.now(timezone.utc)
    played_at = now - timedelta(hours=10)
    try:
        for i in range(3):
            con.execute(
                """
                INSERT INTO songs (
                    track_id, track_name, artist_id, artist_name, spotify_uri, first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    f"track-{i}",
                    f"Track {i}",
                    "artist",
                    "Artist",
                    f"spotify:track:{i}",
                    played_at,
                    now,
                ],
            )
            con.execute(
                """
                INSERT INTO listening_events (
                    event_id, session_id, track_id, track_name, artist_id, artist_name,
                    started_at, ended_at, target_score, was_skipped, created_at
                ) VALUES (?, 's1', ?, ?, 'artist', 'Artist', ?, ?, 0.9, false, ?);
                """,
                [f"e-{i}", f"track-{i}", f"Track {i}", played_at, played_at, now],
            )
    finally:
        con.close()

    settings = Settings(
        DATABASE_PATH=str(db_path),
        MODEL_PATH=str(tmp_path / "missing.keras"),
        QUEUE_BATCH_SIZE=2,
        MIN_CANDIDATE_TARGET_SCORE=0.55,
    ).resolve_paths(tmp_path)
    service = QueueService.queuer(settings)
    client = FakeSpotifyClient()

    service._maybe_queue(client)  # type: ignore[arg-type]

    con = duckdb.connect(str(db_path))
    try:
        action_count = con.execute("SELECT COUNT(*) FROM queue_actions").fetchone()[0]
    finally:
        con.close()

    assert len(client.queued) == 2
    assert action_count == 2

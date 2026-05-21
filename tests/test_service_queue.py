from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.db import migrate
from rubik_spotify_queue.service import QueueService
from rubik_spotify_queue.spotify import SpotifyQueueState


class FakeSpotifyClient:
    def __init__(self, *, upcoming_uris: tuple[str, ...] = ()) -> None:
        self.queued: list[str] = []
        self.upcoming_uris = upcoming_uris

    def current_queue(self) -> SpotifyQueueState:
        return SpotifyQueueState(
            currently_playing_uri="spotify:track:current",
            upcoming_uris=self.upcoming_uris,
            raw_json="{}",
        )

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
        QUEUE_TARGET_BUFFER_SIZE=2,
        CANDIDATE_MIN_TOTAL_PLAYS=1,
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


def test_queue_fills_target_buffer_even_when_batch_size_is_smaller(tmp_path) -> None:
    db_path = tmp_path / "service.duckdb"
    migrate(db_path)
    con = duckdb.connect(str(db_path))
    now = datetime.now(timezone.utc)
    played_at = now - timedelta(hours=10)
    try:
        for i in range(6):
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
        QUEUE_TARGET_BUFFER_SIZE=5,
        CANDIDATE_MIN_TOTAL_PLAYS=1,
        MIN_CANDIDATE_TARGET_SCORE=0.55,
    ).resolve_paths(tmp_path)
    service = QueueService.queuer(settings)
    client = FakeSpotifyClient()

    service._maybe_queue(client)  # type: ignore[arg-type]

    assert len(client.queued) == 5


def test_queue_waits_when_spotify_queue_buffer_is_full(tmp_path) -> None:
    db_path = tmp_path / "service.duckdb"
    migrate(db_path)
    con = duckdb.connect(str(db_path))
    now = datetime.now(timezone.utc)
    try:
        for uri in ("spotify:track:a", "spotify:track:b"):
            con.execute(
                """
                INSERT INTO queue_actions (
                    queue_action_id, created_at, track_id, spotify_uri,
                    predicted_score, model_version, status, detail
                ) VALUES (?, ?, ?, ?, 0.9, 'test', 'queued', '');
                """,
                [uri, now, uri, uri],
            )
    finally:
        con.close()

    settings = Settings(
        DATABASE_PATH=str(db_path),
        MODEL_PATH=str(tmp_path / "missing.keras"),
        QUEUE_BATCH_SIZE=2,
        QUEUE_TARGET_BUFFER_SIZE=2,
        CANDIDATE_MIN_TOTAL_PLAYS=1,
    ).resolve_paths(tmp_path)
    service = QueueService.queuer(settings)
    client = FakeSpotifyClient(upcoming_uris=("spotify:track:a", "spotify:track:b"))

    service._maybe_queue(client)  # type: ignore[arg-type]

    assert client.queued == []


def test_queue_only_fills_open_buffer_slots(tmp_path) -> None:
    db_path = tmp_path / "service.duckdb"
    migrate(db_path)
    con = duckdb.connect(str(db_path))
    now = datetime.now(timezone.utc)
    played_at = now - timedelta(hours=10)
    try:
        con.execute(
            """
            INSERT INTO queue_actions (
                queue_action_id, created_at, track_id, spotify_uri,
                predicted_score, model_version, status, detail
            ) VALUES ('existing-action', ?, 'existing', 'spotify:track:existing', 0.9, 'test', 'queued', '');
            """,
            [now],
        )
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
        QUEUE_TARGET_BUFFER_SIZE=2,
        CANDIDATE_MIN_TOTAL_PLAYS=1,
        MIN_CANDIDATE_TARGET_SCORE=0.55,
    ).resolve_paths(tmp_path)
    service = QueueService.queuer(settings)
    client = FakeSpotifyClient(upcoming_uris=("spotify:track:existing",))

    service._maybe_queue(client)  # type: ignore[arg-type]

    assert len(client.queued) == 1

from __future__ import annotations

import duckdb
import pandas as pd

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.db import migrate
from rubik_spotify_queue.recommender import Candidate, apply_recent_penalty, fallback_scores, model_input_dict
from rubik_spotify_queue.recommender import score_candidates
from rubik_spotify_queue.recommender import select_weighted_queue_batch


def test_fallback_scores_blend_unknowns_toward_half() -> None:
    frame = pd.DataFrame(
        {
            "avg_target_score": [1.0, 0.0, None],
            "plays": [5, 5, 0],
        }
    )
    scores = fallback_scores(frame)
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] == 0.5


def test_recent_penalty_lowers_recent_track_score() -> None:
    frame = pd.DataFrame(
        {
            "predicted_score": [0.8],
            "last_played_at": [pd.Timestamp.now(tz="UTC")],
        }
    )
    out = apply_recent_penalty(frame)
    assert float(out["predicted_score"].iloc[0]) < 0.8


def test_model_input_dict_uses_album_input_contract() -> None:
    frame = pd.DataFrame(
        {
            "track_id": ["spotify:track:1"],
            "artist_id": ["Artist"],
            "album_name": ["Album"],
            "hour_sin": [0.1],
            "hour_cos": [0.2],
            "day_sin": [0.3],
            "day_cos": [0.4],
            "platform": ["ignored"],
            "shuffle_numeric": [1.0],
        }
    )
    inputs = model_input_dict(frame)
    assert sorted(inputs) == [
        "album_name",
        "artist_id",
        "day_cos",
        "day_sin",
        "hour_cos",
        "hour_sin",
        "track_id",
    ]


def test_score_candidates_uses_fallback_when_model_missing(tmp_path) -> None:
    db_path = tmp_path / "service.duckdb"
    migrate(db_path)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            INSERT INTO songs (
                track_id, track_name, artist_id, artist_name, album_name, spotify_uri, first_seen_at, last_seen_at
            ) VALUES ('spotify:track:one', 'One', 'Artist 1', 'Artist 1', 'Album 1', 'spotify:track:one', now(), now());
            """
        )
        con.execute(
            """
            INSERT INTO listening_events (
                event_id, session_id, track_id, track_name, artist_id, artist_name,
                started_at, ended_at, target_score, was_skipped, created_at
            ) VALUES ('e1', 's1', 'spotify:track:one', 'One', 'Artist 1', 'Artist 1',
                now() - INTERVAL 10 HOUR, now() - INTERVAL 10 HOUR, 0.9, false, now());
            """
        )
        settings = Settings(
            MODEL_PATH=str(tmp_path / "missing.keras"),
            CANDIDATE_MIN_TOTAL_PLAYS=1,
        ).resolve_paths(tmp_path)
        candidates = score_candidates(con, settings)
    finally:
        con.close()

    assert len(candidates) == 1
    assert candidates[0].model_version == "history_fallback_v1"
    assert candidates[0].predicted_score > 0.5


def test_weighted_queue_batch_uses_top_pool_without_replacement() -> None:
    candidates = [
        Candidate(
            track_id=f"track-{i}",
            spotify_uri=f"spotify:track:{i}",
            track_name=f"Track {i}",
            artist_id="artist",
            artist_name="Artist",
            predicted_score=score,
            model_version="test",
        )
        for i, score in enumerate([0.95, 0.90, 0.85, 0.80])
    ]
    settings = Settings(
        QUEUE_BATCH_SIZE=2,
        QUEUE_RANDOM_POOL_SIZE=2,
        QUEUE_SCORE_WEIGHT_POWER=4.0,
        MIN_CANDIDATE_TARGET_SCORE=0.55,
    )

    selected = select_weighted_queue_batch(
        candidates,
        settings,
        rng=__import__("numpy").random.default_rng(7),
    )

    assert len(selected) == 2
    assert len({candidate.track_id for candidate in selected}) == 2
    assert {candidate.track_id for candidate in selected}.issubset({"track-0", "track-1"})

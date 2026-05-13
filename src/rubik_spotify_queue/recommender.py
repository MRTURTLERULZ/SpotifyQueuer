"""Candidate generation and model scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.time_features import local_time_parts


@dataclass(frozen=True)
class Candidate:
    track_id: str
    spotify_uri: str
    track_name: str | None
    artist_id: str | None
    artist_name: str | None
    predicted_score: float
    model_version: str


class TensorFlowScorer:
    """Thin adapter for a Colab-exported Keras model.

    The model contract is the six-input Colab export:
    track_id, artist_id, hour_sin, hour_cos, day_sin, day_cos.
    If the model is not present, callers should use the fallback scorer.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Any | None = None

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.available:
            raise FileNotFoundError(self.model_path)
        try:
            from tensorflow import keras  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("TensorFlow is not installed in this environment.") from exc
        self.model = keras.models.load_model(self.model_path)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.load()
        assert self.model is not None
        inputs = model_input_dict(frame)
        raw = self.model.predict(inputs, verbose=0)
        return np.asarray(raw).reshape(-1).astype(float)


def candidate_frame(con: duckdb.DuckDBPyConnection, settings: Settings, *, limit: int = 500) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    tf = local_time_parts(now, settings.timezone)
    rows = con.execute(
        """
        SELECT
            s.track_id,
            s.spotify_uri,
            s.track_name,
            s.artist_id,
            s.artist_name,
            s.last_seen_at,
            AVG(e.target_score) AS avg_target_score,
            COUNT(e.event_id) AS plays,
            MAX(e.ended_at) AS last_played_at
        FROM songs s
        LEFT JOIN listening_events e ON e.track_id = s.track_id
        WHERE s.spotify_uri IS NOT NULL
        GROUP BY s.track_id, s.spotify_uri, s.track_name, s.artist_id, s.artist_name, s.last_seen_at
        ORDER BY COALESCE(MAX(e.ended_at), s.last_seen_at) DESC
        LIMIT ?;
        """,
        [limit],
    ).fetchdf()
    if rows.empty:
        return rows
    for key, value in tf.items():
        rows[key] = value
    return rows


def model_input_dict(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "track_id": frame["track_id"].astype(str).to_numpy(),
        "artist_id": frame["artist_id"].fillna("unknown").astype(str).to_numpy(),
        "hour_sin": frame["hour_sin"].astype("float32").to_numpy(),
        "hour_cos": frame["hour_cos"].astype("float32").to_numpy(),
        "day_sin": frame["day_sin"].astype("float32").to_numpy(),
        "day_cos": frame["day_cos"].astype("float32").to_numpy(),
    }


def score_candidates(con: duckdb.DuckDBPyConnection, settings: Settings, *, limit: int = 500) -> list[Candidate]:
    frame = candidate_frame(con, settings, limit=limit)
    return rank_candidate_frame(frame, settings)


def rank_candidate_frame(frame: pd.DataFrame, settings: Settings) -> list[Candidate]:
    if frame.empty:
        return []

    model_version = "history_fallback_v1"
    scores = fallback_scores(frame)

    scorer = TensorFlowScorer(settings.model_path)
    if scorer.available:
        try:
            scores = scorer.predict(frame)
            model_version = f"tensorflow:{settings.model_path.name}"
        except Exception:
            model_version = "history_fallback_v1_after_tf_error"

    frame = frame.copy()
    frame["predicted_score"] = np.clip(scores, 0.0, 1.0)
    frame = apply_recent_penalty(frame)
    frame = frame.sort_values("predicted_score", ascending=False)

    return [
        Candidate(
            track_id=str(row.track_id),
            spotify_uri=str(row.spotify_uri),
            track_name=str(row.track_name) if pd.notna(row.track_name) else None,
            artist_id=str(row.artist_id) if pd.notna(row.artist_id) else None,
            artist_name=str(row.artist_name) if pd.notna(row.artist_name) else None,
            predicted_score=float(row.predicted_score),
            model_version=model_version,
        )
        for row in frame.itertuples(index=False)
    ]


def select_weighted_queue_batch(
    candidates: list[Candidate],
    settings: Settings,
    *,
    already_queued_track_ids: set[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[Candidate]:
    already = already_queued_track_ids or set()
    eligible = [
        candidate
        for candidate in candidates
        if candidate.predicted_score >= settings.min_candidate_target_score
        and candidate.track_id not in already
    ]
    if not eligible:
        return []

    pool_size = max(1, int(settings.queue_random_pool_size))
    batch_size = max(1, int(settings.queue_batch_size))
    pool = sorted(eligible, key=lambda candidate: candidate.predicted_score, reverse=True)[:pool_size]
    sample_size = min(batch_size, len(pool))

    weights = np.asarray(
        [max(0.0, candidate.predicted_score) ** float(settings.queue_score_weight_power) for candidate in pool],
        dtype=float,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        probabilities = None
    else:
        probabilities = weights / float(weights.sum())

    generator = rng or np.random.default_rng()
    selected_indices = generator.choice(len(pool), size=sample_size, replace=False, p=probabilities)
    return [pool[int(index)] for index in selected_indices]


def fallback_scores(frame: pd.DataFrame) -> np.ndarray:
    avg = pd.to_numeric(frame["avg_target_score"], errors="coerce").fillna(0.5)
    plays = pd.to_numeric(frame["plays"], errors="coerce").fillna(0)
    confidence = np.minimum(plays.to_numpy(dtype=float) / 5.0, 1.0)
    return (confidence * avg.to_numpy(dtype=float)) + ((1.0 - confidence) * 0.5)


def apply_recent_penalty(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    last = pd.to_datetime(out["last_played_at"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    hours_since = (now - last).dt.total_seconds() / 3600.0
    penalty = np.where(hours_since.fillna(9999) < 6, 0.25, 0.0)
    out["predicted_score"] = np.clip(out["predicted_score"].astype(float) - penalty, 0.0, 1.0)
    return out

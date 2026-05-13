"""Spotify Extended Streaming History ingestion for model-ready training data."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rubik_spotify_queue.time_features import continuous_time_parts


MODEL_COLUMNS = [
    "track_id",
    "artist_id",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "target_score",
]

DEBUG_COLUMNS = [
    "ts",
    "local_time",
    "platform",
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
    "album_name",
    "ms_played",
    "skipped",
    "reason_start",
    "reason_end",
    "shuffle",
    "offline",
    "hour",
    "day_of_week",
    "hour_float",
    "day_float",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "target_score",
]

SENSITIVE_COLUMNS = {"ip_addr"}


@dataclass
class IngestStats:
    number_of_source_rows: int = 0
    number_of_source_files: int = 0
    number_of_rows_after_music_metadata: int = 0
    number_of_rows_dropped_missing_track: int = 0
    number_of_rows_dropped_missing_artist: int = 0
    number_of_rows_dropped_incognito: int = 0
    number_of_rows_dropped_low_ms_played: int = 0
    number_of_rows_dropped_trackerror: int = 0
    number_of_rows_dropped_none_target: int = 0
    number_of_rows_final: int = 0
    target_apply_errors: int = 0


@dataclass(frozen=True)
class IngestResult:
    model_ready_dir: Path
    processed_dir: Path
    rows_full: int
    rows_train: int
    rows_val: int
    rows_test: int
    unique_tracks: int
    unique_artists: int


def input_paths_from_dir(raw_history_dir: Path, pattern: str = "Streaming_History*.json") -> list[Path]:
    return sorted(raw_history_dir.glob(pattern))


def ingest_history(
    *,
    raw_history_dir: Path,
    model_ready_dir: Path,
    processed_dir: Path,
    timezone_name: str,
    pattern: str = "Streaming_History*.json",
) -> IngestResult:
    paths = input_paths_from_dir(raw_history_dir, pattern)
    if not paths:
        raise FileNotFoundError(f"No Spotify history JSON files matched {pattern!r} under {raw_history_dir}")

    raw, used_files = load_and_concat_jsons(paths)
    stats = IngestStats(number_of_source_files=len(used_files))
    frame = normalize_columns(raw)
    frame = apply_music_and_quality_filters(frame, stats)
    frame = parse_timestamps_and_local_time(frame, timezone_name)
    frame = compute_continuous_time_features(frame, timezone_name)
    frame = resolve_track_and_artist_id(frame)
    frame = apply_target_scores(frame, stats)

    train, val, test = chronological_split(frame)
    full_model = build_model_frame(frame)
    train_model = build_model_frame(train)
    val_model = build_model_frame(val)
    test_model = build_model_frame(test)
    debug = build_debug_frame(frame)

    write_outputs(
        full_model=full_model,
        train_model=train_model,
        val_model=val_model,
        test_model=test_model,
        debug=debug,
        model_ready_dir=model_ready_dir,
        processed_dir=processed_dir,
        stats=stats,
        used_files=used_files,
        timezone_name=timezone_name,
    )
    return IngestResult(
        model_ready_dir=model_ready_dir,
        processed_dir=processed_dir,
        rows_full=len(full_model),
        rows_train=len(train_model),
        rows_val=len(val_model),
        rows_test=len(test_model),
        unique_tracks=int(full_model["track_id"].nunique()),
        unique_artists=int(full_model["artist_id"].nunique()),
    )


def load_and_concat_jsons(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames = [pd.read_json(path) for path in paths]
    return pd.concat(frames, ignore_index=True), [str(path.resolve()) for path in paths]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def rename_first(canonical: str, variants: list[str]) -> None:
        for variant in variants:
            if variant in out.columns and variant != canonical:
                out.rename(columns={variant: canonical}, inplace=True)
                return

    rename_first("ts", ["ts", "timestamp", "endTime"])
    rename_first("ms_played", ["ms_played", "msPlayed"])
    rename_first("spotify_track_uri", ["spotify_track_uri", "spotifyTrackUri", "track_uri"])
    rename_first("reason_end", ["reason_end", "reasonEnd"])
    rename_first("reason_start", ["reason_start", "reasonStart"])
    rename_first("skipped", ["skipped", "Skipped"])
    rename_first("incognito_mode", ["incognito_mode", "incognitoMode"])
    rename_first("duration_ms", ["duration_ms", "durationMs"])

    if "track_name" in out.columns and "master_metadata_track_name" not in out.columns:
        out.rename(columns={"track_name": "master_metadata_track_name"}, inplace=True)
    if "artist_name" in out.columns and "master_metadata_album_artist_name" not in out.columns:
        out.rename(columns={"artist_name": "master_metadata_album_artist_name"}, inplace=True)

    for column in (
        "spotify_track_uri",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
    ):
        if column not in out.columns:
            out[column] = np.nan

    out.drop(columns=[column for column in SENSITIVE_COLUMNS if column in out.columns], inplace=True)
    return out


def _is_null_like(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        normalized = series.astype(str).str.strip().str.lower()
        return series.isna() | normalized.isin(["", "nan", "none", "<na>"])
    return series.isna()


def _valid_track_uri(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.startswith("spotify:track:", na=False)


def apply_music_and_quality_filters(df: pd.DataFrame, stats: IngestStats) -> pd.DataFrame:
    stats.number_of_source_rows = len(df)
    out = df.copy()

    ep = out.get("episode_name")
    ep_ok = _is_null_like(ep) if ep is not None else pd.Series(True, index=out.index)
    episode_uri = out.get("spotify_episode_uri")
    if episode_uri is not None:
        ep_ok = ep_ok & _is_null_like(episode_uri)
    audiobook = out.get("audiobook_title")
    audiobook_ok = _is_null_like(audiobook) if audiobook is not None else pd.Series(True, index=out.index)
    base = ep_ok & audiobook_ok

    uri_ok = _valid_track_uri(out["spotify_track_uri"])
    track_name_ok = ~_is_null_like(out["master_metadata_track_name"])
    artist_ok = ~_is_null_like(out["master_metadata_album_artist_name"])

    stats.number_of_rows_dropped_missing_track = int((base & ~uri_ok & ~track_name_ok).sum())
    stats.number_of_rows_dropped_missing_artist = int((base & (uri_ok | track_name_ok) & ~artist_ok).sum())
    keep = base & (uri_ok | track_name_ok) & artist_ok
    out = out.loc[keep].copy()
    stats.number_of_rows_after_music_metadata = len(out)

    if "incognito_mode" in out.columns:
        incognito = out["incognito_mode"].map(_coerce_bool)
        stats.number_of_rows_dropped_incognito = int(incognito.sum())
        out = out.loc[~incognito].copy()

    if "ms_played" not in out.columns:
        out["ms_played"] = np.nan
    ms = pd.to_numeric(out["ms_played"], errors="coerce")
    low_ms = ms.isna() | (ms < 5000)
    stats.number_of_rows_dropped_low_ms_played = int(low_ms.sum())
    out = out.loc[~low_ms].copy()
    out["ms_played"] = pd.to_numeric(out["ms_played"], errors="coerce").astype(np.int64)

    if "reason_end" not in out.columns:
        out["reason_end"] = ""
    reason = out["reason_end"].astype(str).str.strip().str.lower()
    stats.number_of_rows_dropped_trackerror = int((reason == "trackerror").sum())
    return out.reset_index(drop=True)


def parse_timestamps_and_local_time(df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.loc[~out["ts"].isna()].copy()
    out["local_time"] = out["ts"].dt.tz_convert(ZoneInfo(timezone_name))
    return out.reset_index(drop=True)


def compute_continuous_time_features(df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    out = df.copy()
    parts = out["ts"].map(lambda value: continuous_time_parts(value.to_pydatetime(), timezone_name))
    out["hour"] = parts.map(lambda value: int(value["hour"]))
    out["day_of_week"] = parts.map(lambda value: int(value["day_of_week"]))
    out["hour_float"] = parts.map(lambda value: float(value["hour_float"]))
    out["day_float"] = parts.map(lambda value: float(value["day_float"]))
    out["hour_sin"] = parts.map(lambda value: float(value["hour_sin"]))
    out["hour_cos"] = parts.map(lambda value: float(value["hour_cos"]))
    out["day_sin"] = parts.map(lambda value: float(value["day_sin"]))
    out["day_cos"] = parts.map(lambda value: float(value["day_cos"]))
    return out


def resolve_track_and_artist_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["track_id"] = [
        _human_track_id_label(track, artist)
        for track, artist in zip(
            out["master_metadata_track_name"],
            out["master_metadata_album_artist_name"],
            strict=True,
        )
    ]
    out["artist_id"] = out["master_metadata_album_artist_name"].astype(str)
    return out


def apply_target_scores(df: pd.DataFrame, stats: IngestStats) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    ms_col = pd.to_numeric(out["ms_played"], errors="coerce")
    skipped_col = out["skipped"] if "skipped" in out.columns else pd.Series(False, index=out.index)
    reason_col = out["reason_end"] if "reason_end" in out.columns else pd.Series("", index=out.index)
    duration_col = out["duration_ms"] if "duration_ms" in out.columns else pd.Series(np.nan, index=out.index)

    scores: list[float | None] = []
    for ms_value, skipped_value, reason_value, duration_value in zip(ms_col, skipped_col, reason_col, duration_col, strict=True):
        try:
            duration_ms = None
            if not pd.isna(duration_value):
                duration_int = int(float(duration_value))
                duration_ms = duration_int if duration_int > 0 else None
            scores.append(compute_listening_target_score(ms_value, skipped_value, reason_value, duration_ms))
        except (TypeError, ValueError):
            stats.target_apply_errors += 1
            scores.append(None)

    out["target_score"] = scores
    none_mask = pd.Series([score is None for score in scores], dtype=bool)
    stats.number_of_rows_dropped_none_target = int(none_mask.sum())
    out = out.loc[~none_mask].copy()
    out["target_score"] = pd.to_numeric(out["target_score"], errors="coerce").clip(0.0, 1.0)
    stats.number_of_rows_final = len(out)
    return out.reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(by=["ts"], ascending=True, kind="mergesort").reset_index(drop=True)
    first = int(math.floor(len(ordered) * 0.8))
    second = int(math.floor(len(ordered) * 0.9))
    return ordered.iloc[:first].copy(), ordered.iloc[first:second].copy(), ordered.iloc[second:].copy()


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "track_id": df["track_id"].astype(str),
            "artist_id": df["artist_id"].astype(str),
            "hour_sin": df["hour_sin"].astype(float),
            "hour_cos": df["hour_cos"].astype(float),
            "day_sin": df["day_sin"].astype(float),
            "day_cos": df["day_cos"].astype(float),
            "target_score": df["target_score"].astype(float),
        }
    )
    return frame[MODEL_COLUMNS]


def build_debug_frame(df: pd.DataFrame) -> pd.DataFrame:
    local_time = df["local_time"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    frame = pd.DataFrame(
        {
            "ts": df["ts"].astype(str),
            "local_time": local_time,
            "platform": df["platform"].fillna("unknown").astype(str) if "platform" in df.columns else "unknown",
            "track_id": df["spotify_track_uri"].astype(str),
            "track_name": df["master_metadata_track_name"].astype(str),
            "artist_id": df["artist_id"].astype(str),
            "artist_name": df["artist_id"].astype(str),
            "album_name": df["master_metadata_album_album_name"].fillna("").astype(str),
            "ms_played": df["ms_played"].astype(int),
            "skipped": df["skipped"].map(_coerce_bool) if "skipped" in df.columns else False,
            "reason_start": df["reason_start"].fillna("").astype(str) if "reason_start" in df.columns else "",
            "reason_end": df["reason_end"].fillna("").astype(str) if "reason_end" in df.columns else "",
            "shuffle": df["shuffle"].map(_coerce_bool) if "shuffle" in df.columns else False,
            "offline": df["offline"].map(_coerce_bool) if "offline" in df.columns else False,
            "hour": df["hour"].astype(int),
            "day_of_week": df["day_of_week"].astype(int),
            "hour_float": df["hour_float"].astype(float),
            "day_float": df["day_float"].astype(float),
            "hour_sin": df["hour_sin"].astype(float),
            "hour_cos": df["hour_cos"].astype(float),
            "day_sin": df["day_sin"].astype(float),
            "day_cos": df["day_cos"].astype(float),
            "target_score": df["target_score"].astype(float),
        }
    )
    return frame[DEBUG_COLUMNS]


def write_outputs(
    *,
    full_model: pd.DataFrame,
    train_model: pd.DataFrame,
    val_model: pd.DataFrame,
    test_model: pd.DataFrame,
    debug: pd.DataFrame,
    model_ready_dir: Path,
    processed_dir: Path,
    stats: IngestStats,
    used_files: list[str],
    timezone_name: str,
) -> None:
    model_ready_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    full_model.to_csv(model_ready_dir / "model_ready_history_full.csv", index=False)
    train_model.to_csv(model_ready_dir / "model_ready_history_train.csv", index=False)
    val_model.to_csv(model_ready_dir / "model_ready_history_val.csv", index=False)
    test_model.to_csv(model_ready_dir / "model_ready_history_test.csv", index=False)
    debug.to_csv(processed_dir / "history_events_processed_debug.csv", index=False)

    write_vocab(sorted(train_model["track_id"].dropna().unique().tolist()), model_ready_dir / "track_vocab.txt")
    write_vocab(sorted(train_model["artist_id"].dropna().unique().tolist()), model_ready_dir / "artist_vocab.txt")

    y = full_model["target_score"].astype(float)
    feature_config = {
        "schema_columns": MODEL_COLUMNS,
        "time_encoding": "continuous_hour_and_day",
        "hour_float": "hour + minute/60 + second/3600 + microsecond/3600000000",
        "day_float": "weekday + hour_float/24; Monday=0",
        "timezone_assumption": timezone_name,
        "input_files": used_files,
        **asdict(stats),
        "train_rows": len(train_model),
        "val_rows": len(val_model),
        "test_rows": len(test_model),
        "full_rows": len(full_model),
        "track_vocab_size": int(train_model["track_id"].nunique()),
        "artist_vocab_size": int(train_model["artist_id"].nunique()),
        "average_target_score": float(y.mean()) if len(y) else 0.0,
    }
    (model_ready_dir / "feature_config.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")


def write_vocab(values: list[str], path: Path) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _human_track_id_label(track_name: object, artist_name: object) -> str:
    track = _clean_label(track_name)
    artist = _clean_label(artist_name)
    if track:
        return track
    if artist:
        return f"Untitled - {artist}"
    return "Unknown track"


def _clean_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _coerce_bool(value: object) -> bool:
    if value is True or value is False:
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def compute_listening_target_score(
    ms_played: object,
    skipped: object,
    reason_end: object,
    duration_ms: int | None = None,
) -> float | None:
    reason = "" if pd.isna(reason_end) else str(reason_end).strip().lower()
    if reason == "trackerror":
        return None
    try:
        ms = int(round(float(ms_played)))
    except (TypeError, ValueError):
        return None
    if ms < 0:
        return None

    user_skipped = _coerce_bool(skipped) or reason in {"fwdbtn", "backbtn"}
    duration = int(duration_ms) if duration_ms and duration_ms > 0 else None
    if duration:
        play_ratio = max(0.0, min(1.0, ms / float(duration)))
        if reason == "trackdone" and not user_skipped:
            raw = max(0.85, play_ratio)
        elif user_skipped:
            raw = _duration_skip_button_score(play_ratio, ms)
        else:
            raw = play_ratio
        return float(max(0.0, min(1.0, raw)))

    if reason == "trackdone" and not user_skipped:
        return 1.0
    if user_skipped:
        return _no_duration_skip_score(ms)
    if reason in {"endplay", "logout", "remote"}:
        return _no_duration_neutral_score(ms)
    return _no_duration_fallback_score(ms)


def _duration_skip_button_score(play_ratio: float, ms_played: int) -> float:
    if play_ratio < 0.10:
        score = 0.0
    elif play_ratio < 0.50:
        score = 0.25 + (play_ratio - 0.10) / 0.40 * 0.25
    elif play_ratio < 0.85:
        score = 0.50 + (play_ratio - 0.50) / 0.35 * 0.25
    else:
        score = 0.85
    if ms_played >= 120_000:
        score = max(score, 0.55)
    return float(max(0.0, min(1.0, score)))


def _no_duration_skip_score(ms_played: int) -> float:
    seconds = ms_played / 1000.0
    if seconds < 15:
        return 0.0
    if seconds < 60:
        return 0.25
    if seconds < 120:
        return 0.45
    return 0.65


def _no_duration_neutral_score(ms_played: int) -> float:
    seconds = ms_played / 1000.0
    if seconds < 30:
        return 0.20
    if seconds < 120:
        return 0.50
    return 0.70


def _no_duration_fallback_score(ms_played: int) -> float:
    seconds = ms_played / 1000.0
    if seconds < 10:
        return 0.0
    if seconds < 30:
        return 0.15
    if seconds < 60:
        return 0.30
    if seconds < 120:
        return 0.55
    return 0.80


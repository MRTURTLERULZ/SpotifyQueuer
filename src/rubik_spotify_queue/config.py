"""Runtime configuration for the Rubik Pi service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SPOTIFY_SCOPES: tuple[str, ...] = (
    "user-read-playback-state",
    "user-read-currently-playing",
    "user-read-recently-played",
    "user-top-read",
    "user-library-read",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-modify-playback-state",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    spotify_client_id: str = Field(default="", validation_alias="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str = Field(default="", validation_alias="SPOTIFY_CLIENT_SECRET")
    spotify_redirect_uri: str = Field(
        default="http://127.0.0.1:8888/callback",
        validation_alias="SPOTIFY_REDIRECT_URI",
    )

    database_path: Path = Field(default=Path("data/rubik_spotify_queue.duckdb"), validation_alias="DATABASE_PATH")
    token_path: Path = Field(default=Path("data/spotify_tokens.json"), validation_alias="TOKEN_PATH")
    model_path: Path = Field(default=Path("models/spotify_skip_model.keras"), validation_alias="MODEL_PATH")
    raw_history_dir: Path = Field(default=Path("data/raw"), validation_alias="RAW_HISTORY_DIR")
    model_ready_dir: Path = Field(default=Path("data/model_ready"), validation_alias="MODEL_READY_DIR")
    processed_dir: Path = Field(default=Path("data/processed"), validation_alias="PROCESSED_DIR")

    timezone: str = Field(default="America/New_York", validation_alias="TIMEZONE")
    location_bucket: str = Field(default="unknown", validation_alias="LOCATION_BUCKET")
    activity_bucket: str = Field(default="unknown", validation_alias="ACTIVITY_BUCKET")

    active_poll_seconds: float = Field(default=8.0, validation_alias="ACTIVE_POLL_SECONDS")
    near_track_end_poll_seconds: float = Field(default=3.0, validation_alias="NEAR_TRACK_END_POLL_SECONDS")
    idle_poll_seconds: float = Field(default=120.0, validation_alias="IDLE_POLL_SECONDS")
    quiet_hours_poll_seconds: float = Field(default=900.0, validation_alias="QUIET_HOURS_POLL_SECONDS")
    rate_limit_max_sleep_seconds: float = Field(default=900.0, validation_alias="RATE_LIMIT_MAX_SLEEP_SECONDS")
    initial_error_backoff_seconds: float = Field(default=5.0, validation_alias="INITIAL_ERROR_BACKOFF_SECONDS")
    max_error_backoff_seconds: float = Field(default=300.0, validation_alias="MAX_ERROR_BACKOFF_SECONDS")
    resource_log_every_n_cycles: int = Field(default=1, validation_alias="RESOURCE_LOG_EVERY_N_CYCLES")
    memory_debug_tracemalloc: bool = Field(default=False, validation_alias="MEMORY_DEBUG_TRACEMALLOC")
    max_event_buffer_snapshots: int = Field(default=720, validation_alias="MAX_EVENT_BUFFER_SNAPSHOTS")
    persist_raw_spotify_payloads: bool = Field(default=False, validation_alias="PERSIST_RAW_SPOTIFY_PAYLOADS")

    queue_start_hour: int = Field(default=7, validation_alias="QUEUE_START_HOUR")
    queue_end_hour: int = Field(default=24, validation_alias="QUEUE_END_HOUR")
    queue_ready_check_seconds: float = Field(default=20.0, validation_alias="QUEUE_READY_CHECK_SECONDS")
    queue_add_cooldown_seconds: float = Field(default=10.0, validation_alias="QUEUE_ADD_COOLDOWN_SECONDS")
    queue_batch_size: int = Field(default=2, validation_alias="QUEUE_BATCH_SIZE")
    queue_target_buffer_size: int = Field(default=2, validation_alias="QUEUE_TARGET_BUFFER_SIZE")
    queue_random_pool_size: int = Field(default=50, validation_alias="QUEUE_RANDOM_POOL_SIZE")
    queue_score_weight_power: float = Field(default=4.0, validation_alias="QUEUE_SCORE_WEIGHT_POWER")
    min_candidate_target_score: float = Field(default=0.55, validation_alias="MIN_CANDIDATE_TARGET_SCORE")
    candidate_min_total_plays: int = Field(default=2, validation_alias="CANDIDATE_MIN_TOTAL_PLAYS")
    queue_track_history_max: int = Field(default=200, validation_alias="QUEUE_TRACK_HISTORY_MAX")
    queue_dry_run: bool = Field(default=False, validation_alias="QUEUE_DRY_RUN")

    spotify_accounts_authorize_url: str = "https://accounts.spotify.com/authorize"
    spotify_token_url: str = "https://accounts.spotify.com/api/token"
    spotify_api_base: str = "https://api.spotify.com/v1"

    @property
    def scope_string(self) -> str:
        return " ".join(SPOTIFY_SCOPES)

    def resolve_paths(self, root: Path | None = None) -> "Settings":
        base = root or Path.cwd()
        for attr in (
            "database_path",
            "token_path",
            "model_path",
            "raw_history_dir",
            "model_ready_dir",
            "processed_dir",
        ):
            value = getattr(self, attr)
            if not value.is_absolute():
                object.__setattr__(self, attr, (base / value).resolve())
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings().resolve_paths(Path.cwd())

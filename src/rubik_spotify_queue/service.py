"""Long-running polling and queueing service."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import tracemalloc
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.db import connect, migrate
from rubik_spotify_queue.eventizer import Eventizer, next_capture_poll_seconds, next_poll_seconds
from rubik_spotify_queue.recommender import (
    Candidate,
    TensorFlowScorer,
    candidate_frame,
    rank_candidate_frame,
    select_weighted_queue_batch,
)
from rubik_spotify_queue.spotify import RateLimited, SpotifyClient
from rubik_spotify_queue.time_features import local_time_parts, within_hour_window

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in minimal installs
    psutil = None  # type: ignore[assignment]


console = Console()

LOCK_RETRY_MESSAGES = (
    "could not set lock",
    "conflicting lock",
    "being used by another process",
)


def _is_lock_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return any(message in detail for message in LOCK_RETRY_MESSAGES)


def runtime_connect(database_path: Path, *, read_only: bool = False):
    last_exc: Exception | None = None
    for _ in range(20):
        try:
            return connect(database_path, read_only=read_only)
        except Exception as exc:
            if not _is_lock_error(exc):
                raise
            last_exc = exc
            time.sleep(0.25)
    assert last_exc is not None
    raise last_exc


def runtime_migrate(database_path: Path) -> None:
    last_exc: Exception | None = None
    for _ in range(20):
        try:
            migrate(database_path)
            return
        except Exception as exc:
            if not _is_lock_error(exc):
                raise
            last_exc = exc
            time.sleep(0.25)
    assert last_exc is not None
    raise last_exc


@dataclass
class ServiceState:
    stop_requested: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    last_queue_at: float = 0.0
    loop_count: int = 0
    poll_count: int = 0
    consecutive_errors: int = 0
    last_loop_duration_seconds: float = 0.0
    last_sleep_seconds: float = 0.0
    last_candidate_count: int = 0
    last_ranked_candidate_count: int = 0
    last_selected_candidate_count: int = 0
    currently_queued_track_ids: set[str] | None = None
    currently_queued_spotify_uris: set[str] | None = None
    queued_track_order: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.currently_queued_track_ids is None:
            self.currently_queued_track_ids = set()
        if self.currently_queued_spotify_uris is None:
            self.currently_queued_spotify_uris = set()


class QueueService:
    def __init__(self, settings: Settings, *, enable_polling: bool = True, enable_queueing: bool = True) -> None:
        self.settings = settings
        self.state = ServiceState()
        self.eventizer = Eventizer(settings)
        self.enable_polling = enable_polling
        self.enable_queueing = enable_queueing
        self.scorer = TensorFlowScorer(settings.model_path)
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        if self._process is not None:
            self._process.cpu_percent(None)
        if self.settings.memory_debug_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)

    def run_forever(self) -> None:
        runtime_migrate(self.settings.database_path)
        client = SpotifyClient(self.settings)
        self._install_signal_handlers()
        console.print(
            f"[green]Rubik Spotify service started[/green] "
            f"polling={self.enable_polling} queueing={self.enable_queueing} "
            f"dry_run={self.settings.queue_dry_run} pid={os.getpid()} "
            f"poll_interval={self.settings.active_poll_seconds:.0f}s "
            f"db={self.settings.database_path}"
        )
        console.print(
            f"[dim]queue config[/dim] target_buffer_size={self.settings.queue_target_buffer_size} "
            f"candidate_min_total_plays={self.settings.candidate_min_total_plays} "
            f"resource_log_every_n_cycles={self.settings.resource_log_every_n_cycles} "
            f"cwd={Path.cwd()}"
        )

        try:
            while not self.state.stop_requested:
                self.state.loop_count += 1
                self._reset_loop_metrics()
                started = time.perf_counter()
                try:
                    sleep_seconds = self.tick(client)
                    self.state.consecutive_errors = 0
                except Exception as exc:
                    self.state.consecutive_errors += 1
                    sleep_seconds = self._error_backoff_seconds()
                    console.print(
                        f"[red]Service loop failed:[/red] {exc}; "
                        f"backing off {sleep_seconds:.0f}s "
                        f"(consecutive_errors={self.state.consecutive_errors})"
                    )
                self.state.last_loop_duration_seconds = time.perf_counter() - started
                self.state.last_sleep_seconds = sleep_seconds
                self._log_resources()
                if self.state.stop_event.wait(max(0.0, sleep_seconds)):
                    break
        finally:
            if self.enable_polling:
                con = runtime_connect(self.settings.database_path)
                try:
                    self.eventizer.finalize(con)
                finally:
                    con.close()
            client.close()
            console.print("[yellow]Service stopped; open event buffer finalized.[/yellow]")

    def tick(self, client: SpotifyClient) -> float:
        now = datetime.now(timezone.utc)
        hour = int(local_time_parts(now, self.settings.timezone)["hour"])
        queue_window_open = within_hour_window(hour, self.settings.queue_start_hour, self.settings.queue_end_hour)

        snapshot = None
        if self.enable_polling:
            self.state.poll_count += 1
            console.print(
                f"[dim]{now.astimezone().strftime('%Y-%m-%d %H:%M:%S')}[/dim] "
                f"[cyan]poll #{self.state.poll_count}[/cyan] checking Spotify playback..."
            )
            try:
                snapshot = client.current_playback()
            except RateLimited as exc:
                console.print(f"[yellow]Spotify rate limited. Sleeping {exc.sleep_seconds:.0f}s.[/yellow]")
                return exc.sleep_seconds
            except Exception as exc:
                console.print(f"[red]Playback poll failed:[/red] {exc}")
                return self.settings.idle_poll_seconds

        if self.enable_polling and snapshot is not None:
            con = runtime_connect(self.settings.database_path)
            try:
                self.eventizer.observe(con, snapshot)
            finally:
                con.close()

        playback_active = snapshot is not None and snapshot.is_playing
        queue_check_allowed = queue_window_open or playback_active
        if self.enable_queueing and queue_check_allowed:
            try:
                self._maybe_queue(client)
            except RateLimited as exc:
                console.print(f"[yellow]Spotify queue check rate limited. Sleeping {exc.sleep_seconds:.0f}s.[/yellow]")
                return exc.sleep_seconds
            except Exception as exc:
                console.print(f"[red]Queue check failed:[/red] {exc}")
                return self.settings.queue_ready_check_seconds

        if self.enable_polling and not self.enable_queueing:
            sleep_seconds = next_capture_poll_seconds(self.settings, snapshot)
            self._log_poll_result(snapshot, sleep_seconds)
            return sleep_seconds
        if self.enable_queueing and not self.enable_polling:
            return self.settings.queue_ready_check_seconds if queue_window_open else self.settings.quiet_hours_poll_seconds
        sleep_seconds = next_poll_seconds(self.settings, snapshot, queue_window_open=queue_window_open)
        if self.enable_polling:
            self._log_poll_result(snapshot, sleep_seconds)
        return sleep_seconds

    @classmethod
    def poller(cls, settings: Settings) -> "QueueService":
        return cls(settings, enable_polling=True, enable_queueing=False)

    @classmethod
    def queuer(cls, settings: Settings) -> "QueueService":
        return cls(settings, enable_polling=False, enable_queueing=True)

    @classmethod
    def combined(cls, settings: Settings) -> "QueueService":
        return cls(settings, enable_polling=True, enable_queueing=True)

    def _maybe_queue(self, client: SpotifyClient) -> None:
        elapsed = time.time() - self.state.last_queue_at
        if elapsed < self.settings.queue_add_cooldown_seconds:
            return

        queue_state = client.current_queue()
        queued_uris = set(queue_state.upcoming_uris)
        occupied_uris = set(queued_uris)
        if queue_state.currently_playing_uri:
            occupied_uris.add(queue_state.currently_playing_uri)

        con = runtime_connect(self.settings.database_path, read_only=True)
        try:
            recent_app_queued_uris = self._recent_queued_spotify_uris(con)
            known_app_queued_uris = (self.state.currently_queued_spotify_uris or set()) | recent_app_queued_uris
            app_buffered_uris = queued_uris & known_app_queued_uris
            self.state.currently_queued_spotify_uris = occupied_uris & known_app_queued_uris
        finally:
            con.close()

        target_buffer = max(1, int(self.settings.queue_target_buffer_size))
        slots_available = target_buffer - len(app_buffered_uris)
        if slots_available <= 0:
            console.print(
                f"[dim]app queue buffer full ({len(app_buffered_uris)}/{target_buffer}; "
                f"Spotify shows {len(queued_uris)} total upcoming); "
                f"checking again in {self.settings.queue_ready_check_seconds:.0f}s[/dim]"
            )
            return

        console.print(
            f"[cyan]queue ready[/cyan] app buffer has {len(app_buffered_uris)}/{target_buffer} "
            f"tracks visible ahead; filling {slots_available} slot(s)."
        )

        con = runtime_connect(self.settings.database_path, read_only=True)
        try:
            frame = candidate_frame(con, self.settings, limit=500)
        finally:
            con.close()

        self.state.last_candidate_count = int(len(frame))
        candidates = rank_candidate_frame(frame, self.settings, scorer=self.scorer)
        self.state.last_ranked_candidate_count = int(len(candidates))
        if not candidates:
            return

        already = self.state.currently_queued_track_ids or set()
        chosen_batch = select_weighted_queue_batch(
            candidates,
            self.settings,
            already_queued_track_ids=already,
            already_queued_spotify_uris=occupied_uris,
            batch_size=slots_available,
        )
        self.state.last_selected_candidate_count = int(len(chosen_batch))
        if not chosen_batch:
            console.print("[yellow]No eligible recommendation candidates available for open queue slots.[/yellow]")
            return

        queued_any = False
        for chosen in chosen_batch:
            status = "queued"
            detail = ""
            try:
                if self.settings.queue_dry_run:
                    code = 0
                    status = "dry_run"
                    detail = "dry_run=true; spotify_status=skipped"
                else:
                    code = client.add_to_queue(chosen.spotify_uri)
                    detail = f"spotify_status={code}"
                self._remember_queued_track(chosen.track_id)
                self.state.currently_queued_spotify_uris.add(chosen.spotify_uri)
                queued_any = True
                console.print(
                    f"[cyan]{'Would queue' if self.settings.queue_dry_run else 'Queued'}[/cyan] "
                    f"{chosen.track_name or chosen.track_id} "
                    f"score={chosen.predicted_score:.3f} model={chosen.model_version}"
                )
            except RateLimited:
                raise
            except Exception as exc:
                status = "error"
                detail = str(exc)[:500]
                console.print(f"[red]Queue failed:[/red] {chosen.track_name or chosen.track_id}: {detail}")

            self._record_queue_action(chosen, status=status, detail=detail)

        if queued_any:
            self.state.last_queue_at = time.time()

    def _remember_queued_track(self, track_id: str) -> None:
        max_items = max(1, int(self.settings.queue_track_history_max))
        tracked = self.state.currently_queued_track_ids or set()
        if track_id in tracked:
            return
        while len(self.state.queued_track_order) >= max_items:
            oldest = self.state.queued_track_order.popleft()
            tracked.discard(oldest)
        tracked.add(track_id)
        self.state.queued_track_order.append(track_id)
        self.state.currently_queued_track_ids = tracked

    def _recent_queued_spotify_uris(self, con) -> set[str]:  # noqa: ANN001
        limit = max(10, int(self.settings.queue_target_buffer_size) * 5)
        rows = con.execute(
            """
            SELECT spotify_uri
            FROM queue_actions
            WHERE status = 'queued' AND spotify_uri IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            [limit],
        ).fetchall()
        return {str(row[0]) for row in rows if row and row[0]}

    def _log_poll_result(self, snapshot, sleep_seconds: float) -> None:
        if snapshot is None:
            console.print(
                f"[dim]poll #{self.state.poll_count}[/dim] no active playback; "
                f"sleeping {sleep_seconds:.0f}s"
            )
            return

        progress = ""
        if snapshot.progress_ms is not None and snapshot.duration_ms:
            progress = f" {snapshot.progress_ms // 1000}s/{snapshot.duration_ms // 1000}s"
        state = "playing" if snapshot.is_playing else "paused"
        track = snapshot.track_name or snapshot.track_id or "unknown track"
        artist = f" - {snapshot.artist_name}" if snapshot.artist_name else ""
        console.print(
            f"[dim]poll #{self.state.poll_count}[/dim] {state}: "
            f"[bold]{track}[/bold]{artist}{progress}; sleeping {sleep_seconds:.0f}s"
        )

    def _reset_loop_metrics(self) -> None:
        self.state.last_candidate_count = 0
        self.state.last_ranked_candidate_count = 0
        self.state.last_selected_candidate_count = 0

    def _error_backoff_seconds(self) -> float:
        base = max(1.0, float(self.settings.initial_error_backoff_seconds))
        cap = max(base, float(self.settings.max_error_backoff_seconds))
        return min(cap, base * (2 ** max(0, self.state.consecutive_errors - 1)))

    def _log_resources(self) -> None:
        every = max(1, int(self.settings.resource_log_every_n_cycles))
        if self.state.loop_count % every != 0:
            return

        rss_mb = None
        cpu_percent = None
        if self._process is not None:
            info = self._process.memory_info()
            rss_mb = info.rss / (1024 * 1024)
            cpu_percent = self._process.cpu_percent(None)

        tracemalloc_detail = ""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_detail = (
                f" tracemalloc_current_mb={current / (1024 * 1024):.1f} "
                f"peak_mb={peak / (1024 * 1024):.1f}"
            )

        rss_text = f"{rss_mb:.1f}" if rss_mb is not None else "unavailable"
        cpu_text = f"{cpu_percent:.1f}" if cpu_percent is not None else "unavailable"
        console.print(
            f"[dim]resources[/dim] loop={self.state.loop_count} rss_mb={rss_text} "
            f"cpu_percent={cpu_text} duration_s={self.state.last_loop_duration_seconds:.2f} "
            f"sleep_s={self.state.last_sleep_seconds:.0f} "
            f"event_buffer={len(self.eventizer.buffer)} "
            f"queued_track_history={len(self.state.currently_queued_track_ids or set())} "
            f"queued_uri_cache={len(self.state.currently_queued_spotify_uris or set())} "
            f"candidates={self.state.last_candidate_count} ranked={self.state.last_ranked_candidate_count} "
            f"selected={self.state.last_selected_candidate_count}"
            f"{tracemalloc_detail}"
        )

    def _record_queue_action(self, candidate: Candidate, *, status: str, detail: str) -> None:
        con = runtime_connect(self.settings.database_path)
        try:
            con.execute(
                """
                INSERT INTO queue_actions (
                    queue_action_id, created_at, track_id, spotify_uri, predicted_score,
                    model_version, status, detail
                ) VALUES (?,?,?,?,?,?,?,?);
                """,
                [
                    str(uuid.uuid4()),
                    datetime.now(timezone.utc),
                    candidate.track_id,
                    candidate.spotify_uri,
                    candidate.predicted_score,
                    candidate.model_version,
                    status,
                    detail,
                ],
            )
        finally:
            con.close()

    def _install_signal_handlers(self) -> None:
        def request_stop(signum, frame) -> None:  # noqa: ANN001
            _ = signum
            _ = frame
            self.state.stop_requested = True
            self.state.stop_event.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)


def health_json(settings: Settings) -> str:
    runtime_migrate(settings.database_path)
    con = runtime_connect(settings.database_path, read_only=True)
    try:
        row = con.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM listening_events),
                (SELECT COUNT(*) FROM songs),
                (SELECT COUNT(*) FROM songs WHERE spotify_uri IS NOT NULL),
                (SELECT COUNT(*) FROM queue_actions);
            """
        ).fetchone()
    finally:
        con.close()
    return json.dumps(
        {
            "database": str(settings.database_path),
            "listening_events": int(row[0]),
            "songs": int(row[1]),
            "queueable_songs": int(row[2]),
            "queue_actions": int(row[3]),
            "model_path_exists": settings.model_path.exists(),
            "queue_target_buffer_size": settings.queue_target_buffer_size,
            "cwd": str(Path.cwd()),
        },
        indent=2,
    )

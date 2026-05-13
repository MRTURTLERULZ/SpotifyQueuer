"""Long-running polling and queueing service."""

from __future__ import annotations

import json
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from rich.console import Console

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.db import connect, migrate
from rubik_spotify_queue.eventizer import Eventizer, next_poll_seconds
from rubik_spotify_queue.recommender import Candidate, score_candidates
from rubik_spotify_queue.spotify import RateLimited, SpotifyClient
from rubik_spotify_queue.time_features import local_time_parts, within_hour_window


console = Console()


@dataclass
class ServiceState:
    stop_requested: bool = False
    last_queue_at: float = 0.0
    currently_queued_track_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.currently_queued_track_ids is None:
            self.currently_queued_track_ids = set()


class QueueService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = ServiceState()
        self.eventizer = Eventizer(settings)

    def run_forever(self) -> None:
        migrate(self.settings.database_path)
        client = SpotifyClient(self.settings)
        self._install_signal_handlers()
        console.print(f"[green]Rubik Spotify queue service started[/green] db={self.settings.database_path}")

        try:
            while not self.state.stop_requested:
                sleep_seconds = self.tick(client)
                time.sleep(sleep_seconds)
        finally:
            with connect(self.settings.database_path) as con:
                self.eventizer.finalize(con)
            client.close()
            console.print("[yellow]Service stopped; open event buffer finalized.[/yellow]")

    def tick(self, client: SpotifyClient) -> float:
        now = datetime.now(timezone.utc)
        hour = int(local_time_parts(now, self.settings.timezone)["hour"])
        queue_window_open = within_hour_window(hour, self.settings.queue_start_hour, self.settings.queue_end_hour)

        try:
            snapshot = client.current_playback()
        except RateLimited as exc:
            console.print(f"[yellow]Spotify rate limited. Sleeping {exc.sleep_seconds:.0f}s.[/yellow]")
            return exc.sleep_seconds
        except Exception as exc:
            console.print(f"[red]Playback poll failed:[/red] {exc}")
            return self.settings.idle_poll_seconds

        with connect(self.settings.database_path) as con:
            if snapshot is not None:
                self.eventizer.observe(con, snapshot)
            if queue_window_open:
                self._maybe_queue(con, client)

        return next_poll_seconds(self.settings, snapshot, queue_window_open=queue_window_open)

    def _maybe_queue(self, con, client: SpotifyClient) -> None:
        elapsed = time.time() - self.state.last_queue_at
        if elapsed < self.settings.min_queue_interval_seconds:
            return

        candidates = score_candidates(con, self.settings, limit=500)
        if not candidates:
            return

        already = self.state.currently_queued_track_ids or set()
        chosen = next(
            (
                candidate
                for candidate in candidates
                if candidate.predicted_score >= self.settings.min_candidate_target_score
                and candidate.track_id not in already
            ),
            None,
        )
        if chosen is None:
            return

        status = "queued"
        detail = ""
        try:
            code = client.add_to_queue(chosen.spotify_uri)
            detail = f"spotify_status={code}"
            already.add(chosen.track_id)
            self.state.last_queue_at = time.time()
            console.print(
                f"[cyan]Queued[/cyan] {chosen.track_name or chosen.track_id} "
                f"score={chosen.predicted_score:.3f} model={chosen.model_version}"
            )
        except Exception as exc:
            status = "error"
            detail = str(exc)[:500]
            console.print(f"[red]Queue failed:[/red] {detail}")

        self._record_queue_action(con, chosen, status=status, detail=detail)

    def _record_queue_action(self, con, candidate: Candidate, *, status: str, detail: str) -> None:
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

    def _install_signal_handlers(self) -> None:
        def request_stop(signum, frame) -> None:  # noqa: ANN001
            _ = signum
            _ = frame
            self.state.stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)


def health_json(settings: Settings) -> str:
    migrate(settings.database_path)
    with connect(settings.database_path, read_only=True) as con:
        row = con.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM listening_events),
                (SELECT COUNT(*) FROM songs),
                (SELECT COUNT(*) FROM songs WHERE spotify_uri IS NOT NULL),
                (SELECT COUNT(*) FROM queue_actions);
            """
        ).fetchone()
    return json.dumps(
        {
            "database": str(settings.database_path),
            "listening_events": int(row[0]),
            "songs": int(row[1]),
            "queueable_songs": int(row[2]),
            "queue_actions": int(row[3]),
            "model_path_exists": settings.model_path.exists(),
        },
        indent=2,
    )

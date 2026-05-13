"""Convert polling snapshots into durable listening events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import duckdb

from rubik_spotify_queue.config import Settings
from rubik_spotify_queue.spotify import PlaybackSnapshot, snapshot_row
from rubik_spotify_queue.time_features import local_time_parts


NEW_SESSION_GAP_SECONDS = 30 * 60


@dataclass
class Eventizer:
    settings: Settings
    buffer: list[PlaybackSnapshot] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    previous_track_id: str | None = None
    previous_event_id: str | None = None
    previous_event_ended_at: datetime | None = None

    def observe(self, con: duckdb.DuckDBPyConnection, snapshot: PlaybackSnapshot) -> None:
        con.execute(
            """
            INSERT INTO snapshots (
                snapshot_id, observed_at, track_id, track_name, artist_id, artist_name,
                album_id, album_name, duration_ms, progress_ms, is_playing,
                device_id, device_type, shuffle_state, context_uri, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
            """,
            snapshot_row(snapshot),
        )

        if not snapshot.track_id:
            return
        if not self.buffer:
            self.buffer.append(snapshot)
            return
        if self.buffer[0].track_id == snapshot.track_id:
            self.buffer.append(snapshot)
            return
        self.finalize(con, next_track_id=snapshot.track_id)
        self.buffer.append(snapshot)

    def finalize(self, con: duckdb.DuckDBPyConnection, *, next_track_id: str | None = None) -> str | None:
        if not self.buffer or not self.buffer[0].track_id:
            self.buffer.clear()
            return None

        snaps = sorted(self.buffer, key=lambda snap: snap.observed_at)
        first = snaps[0]
        last = snaps[-1]
        started_at = first.observed_at.astimezone(timezone.utc)
        ended_at = last.observed_at.astimezone(timezone.utc)

        if self.previous_event_ended_at:
            gap = (started_at - self.previous_event_ended_at.astimezone(timezone.utc)).total_seconds()
            if gap > NEW_SESSION_GAP_SECONDS:
                self.session_id = str(uuid.uuid4())
                self.previous_track_id = None
                self.previous_event_id = None

        duration_ms = max((s.duration_ms or 0 for s in snaps), default=0) or None
        max_progress_ms = max((s.progress_ms or 0 for s in snaps), default=0) or None
        wall_clock_ms = max(1, int((ended_at - started_at).total_seconds() * 1000))
        completion_ratio = None
        if duration_ms and max_progress_ms is not None:
            completion_ratio = max(0.0, min(1.0, max_progress_ms / float(duration_ms)))
        was_skipped = bool(completion_ratio is not None and completion_ratio < 0.35)
        target_score = compute_target_score(completion_ratio, was_skipped)
        tf = local_time_parts(started_at, self.settings.timezone)

        event_id = str(uuid.uuid4())
        con.execute(
            """
            INSERT INTO listening_events (
                event_id, session_id, track_id, track_name, artist_id, artist_name,
                album_id, album_name, started_at, ended_at, duration_ms, max_progress_ms,
                wall_clock_ms, completion_ratio, target_score, was_skipped,
                hour, day_of_week, hour_sin, hour_cos, day_sin, day_cos,
                platform, shuffle_state, location_bucket, activity_bucket,
                previous_track_id, next_track_id, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
            """,
            [
                event_id,
                self.session_id,
                first.track_id,
                first.track_name,
                first.artist_id,
                first.artist_name,
                first.album_id,
                first.album_name,
                started_at,
                ended_at,
                duration_ms,
                max_progress_ms,
                wall_clock_ms,
                completion_ratio,
                target_score,
                was_skipped,
                int(tf["hour"]),
                int(tf["day_of_week"]),
                float(tf["hour_sin"]),
                float(tf["hour_cos"]),
                float(tf["day_sin"]),
                float(tf["day_cos"]),
                last.device_type or "unknown",
                last.shuffle_state,
                self.settings.location_bucket,
                self.settings.activity_bucket,
                self.previous_track_id,
                next_track_id,
                datetime.now(timezone.utc),
            ],
        )
        self._merge_song(con, last)
        if self.previous_event_id:
            con.execute("UPDATE listening_events SET next_track_id = ? WHERE event_id = ?;", [first.track_id, self.previous_event_id])

        self.previous_event_id = event_id
        self.previous_track_id = first.track_id
        self.previous_event_ended_at = ended_at
        self.buffer.clear()
        return event_id

    def _merge_song(self, con: duckdb.DuckDBPyConnection, snapshot: PlaybackSnapshot) -> None:
        con.execute(
            """
            INSERT INTO songs (
                track_id, track_name, artist_id, artist_name, album_id, album_name,
                duration_ms, spotify_uri, history_play_count, history_avg_target_score,
                first_seen_at, last_seen_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (track_id) DO UPDATE SET
                track_name = COALESCE(EXCLUDED.track_name, songs.track_name),
                artist_id = COALESCE(EXCLUDED.artist_id, songs.artist_id),
                artist_name = COALESCE(EXCLUDED.artist_name, songs.artist_name),
                album_id = COALESCE(EXCLUDED.album_id, songs.album_id),
                album_name = COALESCE(EXCLUDED.album_name, songs.album_name),
                duration_ms = COALESCE(EXCLUDED.duration_ms, songs.duration_ms),
                spotify_uri = COALESCE(EXCLUDED.spotify_uri, songs.spotify_uri),
                last_seen_at = EXCLUDED.last_seen_at;
            """,
            [
                snapshot.track_id,
                snapshot.track_name,
                snapshot.artist_id,
                snapshot.artist_name,
                snapshot.album_id,
                snapshot.album_name,
                snapshot.duration_ms,
                snapshot.spotify_uri,
                None,
                None,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ],
        )


def compute_target_score(completion_ratio: float | None, was_skipped: bool) -> float | None:
    if completion_ratio is None:
        return None
    score = completion_ratio - (0.20 if was_skipped else 0.0)
    return max(0.0, min(1.0, score))


def next_poll_seconds(settings: Settings, snapshot: PlaybackSnapshot | None, *, queue_window_open: bool) -> float:
    if not queue_window_open:
        return settings.quiet_hours_poll_seconds
    return next_capture_poll_seconds(settings, snapshot)


def next_capture_poll_seconds(settings: Settings, snapshot: PlaybackSnapshot | None) -> float:
    if snapshot is None or not snapshot.is_playing:
        return settings.idle_poll_seconds
    if snapshot.duration_ms and snapshot.progress_ms is not None:
        remaining = snapshot.duration_ms - snapshot.progress_ms
        if 0 < remaining <= 30_000:
            return settings.near_track_end_poll_seconds
    return settings.active_poll_seconds


def event_count(con: duckdb.DuckDBPyConnection) -> int:
    return int(con.execute("SELECT COUNT(*) FROM listening_events;").fetchone()[0])


def summary(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    row = con.execute(
        """
        SELECT COUNT(*), AVG(target_score), AVG(CAST(was_skipped AS DOUBLE))
        FROM listening_events;
        """
    ).fetchone()
    return {
        "listening_events": int(row[0] or 0),
        "avg_target_score": float(row[1]) if row[1] is not None else None,
        "skip_rate": float(row[2]) if row[2] is not None else None,
    }

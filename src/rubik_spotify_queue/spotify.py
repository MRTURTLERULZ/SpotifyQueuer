"""Spotify auth, playback normalization, and API calls."""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from rubik_spotify_queue.config import Settings


class RateLimited(RuntimeError):
    def __init__(self, sleep_seconds: float) -> None:
        super().__init__(f"Spotify rate limited; sleep {sleep_seconds:.1f}s")
        self.sleep_seconds = sleep_seconds


@dataclass(frozen=True)
class PlaybackSnapshot:
    observed_at: datetime
    track_id: str | None
    track_name: str | None
    artist_id: str | None
    artist_name: str | None
    album_id: str | None
    album_name: str | None
    duration_ms: int | None
    progress_ms: int | None
    is_playing: bool
    device_id: str | None
    device_type: str | None
    shuffle_state: bool | None
    context_uri: str | None
    spotify_uri: str | None
    raw_json: str


@dataclass(frozen=True)
class SpotifyQueueState:
    currently_playing_uri: str | None
    upcoming_uris: tuple[str, ...]
    raw_json: str


class TokenStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_login(settings: Settings) -> None:
    if not settings.spotify_client_id or not settings.spotify_client_secret:
        raise ValueError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env first.")

    parsed = urlparse(settings.spotify_redirect_uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8888
    state = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=24))
    result: dict[str, str | None] = {"code": None, "state": None, "error": None}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            params = parse_qs(urlparse(self.path).query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            result["error"] = params.get("error", [None])[0]
            body = b"<html><body><p>Login complete. Return to the terminal.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

    server = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    query = urlencode({
        'client_id': settings.spotify_client_id,
        'response_type': 'code',
        'redirect_uri': settings.spotify_redirect_uri,
        'scope': settings.scope_string,
        'state': state,
    })
    auth_url = f"{settings.spotify_accounts_authorize_url}?{query}"
    webbrowser.open(auth_url)

    try:
        if not done.wait(300):
            raise TimeoutError("Spotify login timed out waiting for callback.")
    finally:
        server.shutdown()

    if result["error"]:
        raise RuntimeError(f"Spotify OAuth error: {result['error']}")
    if result["state"] != state:
        raise RuntimeError("Spotify OAuth state mismatch.")
    if not result["code"]:
        raise RuntimeError("Spotify OAuth callback did not include a code.")

    with httpx.Client(timeout=30) as client:
        response = client.post(
            settings.spotify_token_url,
            data={
                "grant_type": "authorization_code",
                "code": result["code"],
                "redirect_uri": settings.spotify_redirect_uri,
                "client_id": settings.spotify_client_id,
                "client_secret": settings.spotify_client_secret,
            },
        )
        response.raise_for_status()
        tokens = response.json()

    TokenStore(settings.token_path).save(
        {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token", ""),
            "expires_at": time.time() + float(tokens.get("expires_in", 3600)),
            "scope": tokens.get("scope", settings.scope_string),
        }
    )


class SpotifyClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.token_store = TokenStore(settings.token_path)
        self.client = httpx.Client(base_url=settings.spotify_api_base, timeout=30)

    def close(self) -> None:
        self.client.close()

    def _refresh(self, refresh_token: str) -> dict[str, Any]:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                self.settings.spotify_token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.settings.spotify_client_id,
                    "client_secret": self.settings.spotify_client_secret,
                },
            )
            response.raise_for_status()
            return response.json()

    def _access_token(self) -> str:
        data = self.token_store.load()
        if not data:
            raise RuntimeError(f"Missing Spotify tokens at {self.settings.token_path}; run login.")
        if time.time() > float(data.get("expires_at", 0)) - 60:
            refresh = str(data.get("refresh_token") or "")
            if not refresh:
                raise RuntimeError("Spotify token expired and no refresh token is available.")
            fresh = self._refresh(refresh)
            data["access_token"] = fresh["access_token"]
            data["expires_at"] = time.time() + float(fresh.get("expires_in", 3600))
            if fresh.get("refresh_token"):
                data["refresh_token"] = fresh["refresh_token"]
            self.token_store.save(data)
        return str(data["access_token"])

    def request(self, method: str, path: str, *, params: dict[str, str] | None = None) -> httpx.Response:
        token = self._access_token()
        response = self.client.request(method, path, headers={"Authorization": f"Bearer {token}"}, params=params)
        if response.status_code == 401:
            token = self._access_token()
            response = self.client.request(method, path, headers={"Authorization": f"Bearer {token}"}, params=params)
        if response.status_code == 429:
            raw = response.headers.get("retry-after")
            try:
                requested = float(raw) if raw is not None else 60.0
            except ValueError:
                requested = 60.0
            raise RateLimited(min(max(requested, 1.0), self.settings.rate_limit_max_sleep_seconds))
        return response

    def current_playback(self) -> PlaybackSnapshot | None:
        response = self.request("GET", "/me/player/currently-playing")
        if response.status_code in (202, 204) or not response.content:
            return None
        response.raise_for_status()
        return normalize_playback(response.json(), observed_at=datetime.now(timezone.utc))

    def current_queue(self) -> SpotifyQueueState:
        response = self.request("GET", "/me/player/queue")
        if response.status_code in (202, 204) or not response.content:
            return SpotifyQueueState(currently_playing_uri=None, upcoming_uris=(), raw_json="")
        response.raise_for_status()
        return normalize_queue(response.json())

    def add_to_queue(self, spotify_uri: str) -> int:
        response = self.request("POST", "/me/player/queue", params={"uri": spotify_uri})
        if response.status_code not in (200, 204):
            response.raise_for_status()
        return response.status_code


def normalize_playback(body: dict[str, Any], *, observed_at: datetime) -> PlaybackSnapshot | None:
    item = body.get("item")
    if not isinstance(item, dict):
        return None
    artists = item.get("artists") if isinstance(item.get("artists"), list) else []
    first_artist = artists[0] if artists and isinstance(artists[0], dict) else {}
    album = item.get("album") if isinstance(item.get("album"), dict) else {}
    device = body.get("device") if isinstance(body.get("device"), dict) else {}
    context = body.get("context") if isinstance(body.get("context"), dict) else {}

    return PlaybackSnapshot(
        observed_at=observed_at.astimezone(timezone.utc),
        track_id=str(item["id"]) if item.get("id") else None,
        track_name=str(item["name"]) if item.get("name") else None,
        artist_id=str(first_artist["id"]) if first_artist.get("id") else None,
        artist_name=str(first_artist["name"]) if first_artist.get("name") else None,
        album_id=str(album["id"]) if album.get("id") else None,
        album_name=str(album["name"]) if album.get("name") else None,
        duration_ms=int(item["duration_ms"]) if isinstance(item.get("duration_ms"), int) else None,
        progress_ms=int(body["progress_ms"]) if isinstance(body.get("progress_ms"), int) else None,
        is_playing=bool(body.get("is_playing", False)),
        device_id=str(device["id"]) if device.get("id") else None,
        device_type=str(device["type"]) if device.get("type") else None,
        shuffle_state=body.get("shuffle_state") if isinstance(body.get("shuffle_state"), bool) else None,
        context_uri=str(context["uri"]) if context.get("uri") else None,
        spotify_uri=str(item["uri"]) if item.get("uri") else None,
        raw_json=json.dumps(body, separators=(",", ":")),
    )


def normalize_queue(body: dict[str, Any]) -> SpotifyQueueState:
    currently_playing = body.get("currently_playing")
    current_uri = None
    if isinstance(currently_playing, dict) and currently_playing.get("uri"):
        current_uri = str(currently_playing["uri"])

    upcoming: list[str] = []
    raw_queue = body.get("queue")
    if isinstance(raw_queue, list):
        for item in raw_queue:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if uri:
                upcoming.append(str(uri))

    return SpotifyQueueState(
        currently_playing_uri=current_uri,
        upcoming_uris=tuple(upcoming),
        raw_json=json.dumps(body, separators=(",", ":")),
    )


def snapshot_row(snapshot: PlaybackSnapshot) -> list[Any]:
    return [
        str(uuid.uuid4()),
        snapshot.observed_at,
        snapshot.track_id,
        snapshot.track_name,
        snapshot.artist_id,
        snapshot.artist_name,
        snapshot.album_id,
        snapshot.album_name,
        snapshot.duration_ms,
        snapshot.progress_ms,
        snapshot.is_playing,
        snapshot.device_id,
        snapshot.device_type,
        snapshot.shuffle_state,
        snapshot.context_uri,
        snapshot.raw_json,
    ]

"""Command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from rubik_spotify_queue.config import get_settings
from rubik_spotify_queue.db import connect, migrate
from rubik_spotify_queue.history_ingest import ingest_history
from rubik_spotify_queue.history_seed import seed_songs_from_model_ready
from rubik_spotify_queue.model_training import train_and_save
from rubik_spotify_queue.service import QueueService, health_json
from rubik_spotify_queue.spotify import run_login


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rubik Pi Spotify queue service")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="Create or migrate the DuckDB database")
    sub.add_parser("login", help="Run Spotify OAuth and save local tokens")
    sub.add_parser("poll", help="Run the always-on playback polling service only")
    p_queue = sub.add_parser("queue", help="Run the queueing service only")
    p_queue.add_argument("--dry-run", action="store_true", help="Score and log queue choices without adding songs")
    p_serve = sub.add_parser("serve", help="Run combined polling and queueing in one process")
    p_serve.add_argument("--dry-run", action="store_true", help="Score and log queue choices without adding songs")
    sub.add_parser("health", help="Print service/database health JSON")

    p_ingest = sub.add_parser("ingest-history", help="Build model-ready data from Spotify history JSON exports")
    p_ingest.add_argument("--raw-history-dir", default=None, help="Directory containing Streaming_History*.json")
    p_ingest.add_argument("--model-ready-dir", default=None, help="Directory for generated model-ready CSVs")
    p_ingest.add_argument("--processed-dir", default=None, help="Directory for generated debug/processed CSVs")
    p_ingest.add_argument(
        "--live-events-db-path",
        default=None,
        help="Optional DuckDB with live listening_events to append",
    )
    p_ingest.add_argument("--pattern", default="Streaming_History*.json")

    p_seed = sub.add_parser("seed-history", help="Seed songs from model-ready history CSVs")
    p_seed.add_argument("--model-ready-dir", default=None, help="Directory containing model_ready_history_full.csv")

    p_train = sub.add_parser("train-model", help="Train and save the TensorFlow recommender")
    p_train.add_argument("--model-ready-dir", default=None, help="Directory containing train/val/test model-ready CSVs")
    p_train.add_argument("--model-path", default=None, help="Output .keras model path")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.command == "init-db":
        migrate(settings.database_path)
        console.print(f"[green]Database ready:[/green] {settings.database_path}")
    elif args.command == "login":
        run_login(settings)
        console.print(f"[green]Spotify tokens saved:[/green] {settings.token_path}")
    elif args.command == "poll":
        QueueService.poller(settings).run_forever()
    elif args.command == "queue":
        if args.dry_run:
            object.__setattr__(settings, "queue_dry_run", True)
        QueueService.queuer(settings).run_forever()
    elif args.command == "serve":
        if args.dry_run:
            object.__setattr__(settings, "queue_dry_run", True)
        QueueService.combined(settings).run_forever()
    elif args.command == "health":
        console.print(health_json(settings))
    elif args.command == "ingest-history":
        raw_history_dir = settings.raw_history_dir if args.raw_history_dir is None else Path(args.raw_history_dir).resolve()
        model_ready_dir = settings.model_ready_dir if args.model_ready_dir is None else Path(args.model_ready_dir).resolve()
        processed_dir = settings.processed_dir if args.processed_dir is None else Path(args.processed_dir).resolve()
        result = ingest_history(
            raw_history_dir=raw_history_dir,
            model_ready_dir=model_ready_dir,
            processed_dir=processed_dir,
            timezone_name=settings.timezone,
            pattern=args.pattern,
            live_events_db_path=None
            if args.live_events_db_path is None
            else Path(args.live_events_db_path).resolve(),
        )
        console.print(
            f"[green]Ingested Spotify history:[/green] rows full/train/val/test="
            f"{result.rows_full}/{result.rows_train}/{result.rows_val}/{result.rows_test}; "
            f"unique tracks={result.unique_tracks}; unique artists={result.unique_artists}; "
            f"unique albums={result.unique_albums}"
        )
    elif args.command == "seed-history":
        migrate(settings.database_path)
        model_ready_dir = settings.model_ready_dir if args.model_ready_dir is None else Path(args.model_ready_dir).resolve()
        with connect(settings.database_path) as con:
            result = seed_songs_from_model_ready(con, model_ready_dir)
        console.print(
            f"[green]Seeded history songs:[/green] {result.songs_upserted} "
            f"unique track/artist pairs from {result.rows_read} rows; "
            f"{result.queueable_songs} have Spotify queue URIs"
        )
    elif args.command == "train-model":
        model_ready_dir = settings.model_ready_dir if args.model_ready_dir is None else Path(args.model_ready_dir).resolve()
        model_path = settings.model_path if args.model_path is None else Path(args.model_path).resolve()
        try:
            result = train_and_save(
                model_ready_dir=model_ready_dir,
                model_path=model_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        except ImportError as exc:
            raise SystemExit(
                "TensorFlow is not installed for this Python. Use Python 3.11 or 3.12 and install "
                'with: python -m pip install -e ".[tensorflow]"'
            ) from exc
        console.print(f"[green]Saved model:[/green] {result.model_path}")
        console.print(
            f"rows train/val/test={result.train_rows}/{result.val_rows}/{result.test_rows} "
            f"test_loss={result.test_loss:.4f} test_mae={result.test_mae:.4f}"
        )


if __name__ == "__main__":
    main()

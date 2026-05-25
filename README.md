# Rubik Spotify Queue

Codex-written service for a Rubik Pi 3 that:

1. Polls Spotify playback politely and records finalized listening events.
2. Scores known songs with a TensorFlow model when available.
3. Samples from high-scoring songs and adds a small batch to your Spotify queue.

The old `music-ai-recommender` project is kept as reference. This project is the cleaner service target.

## Design

- `snapshots` stores currently-playing observations. Raw Spotify JSON is not persisted by default; set `PERSIST_RAW_SPOTIFY_PAYLOADS=true` only while debugging.
- `listening_events` stores one row per finished track.
- `songs` stores the local candidate universe.
- `queue_actions` records every queue attempt.

Polling is adaptive:

- `ACTIVE_POLL_SECONDS`: normal playback polling.
- `NEAR_TRACK_END_POLL_SECONDS`: faster checks near the end of a song.
- `IDLE_POLL_SECONDS`: slower checks when nothing is playing.
- `QUIET_HOURS_POLL_SECONDS`: very slow checks outside the configured queue window when nothing is playing.
- Spotify `429` responses sleep according to `Retry-After`, capped by `RATE_LIMIT_MAX_SLEEP_SECONDS`.

## Setup

```powershell
cd rubik-spotify-queue
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env`, then:

```powershell
rubik-spotify-queue init-db
rubik-spotify-queue ingest-history
rubik-spotify-queue seed-history
rubik-spotify-queue login
rubik-spotify-queue poll
```

On a Rubik Pi/Linux shell:

```bash
cd rubik-spotify-queue
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m rubik_spotify_queue.cli init-db
python -m rubik_spotify_queue.cli ingest-history
python -m rubik_spotify_queue.cli seed-history
python -m rubik_spotify_queue.cli train-model
python -m rubik_spotify_queue.cli poll
```

For TensorFlow training/prediction, use Python 3.11 or 3.12 and install the optional extra:

```bash
python -m pip install -e ".[tensorflow]"
```

If TensorFlow does not publish a wheel for your Rubik Pi OS/CPU combination, train on another Python 3.11/3.12 machine and copy `models/spotify_skip_model.keras` to the Pi.

## TensorFlow Model Contract

Train and save the Keras model with:

```bash
python -m rubik_spotify_queue.cli train-model
```

By default, this reads:

```text
data/model_ready/model_ready_history_train.csv
data/model_ready/model_ready_history_val.csv
data/model_ready/model_ready_history_test.csv
```

and writes:

```text
models/spotify_skip_model.keras
```

You can override either path:

```text
MODEL_READY_DIR=/absolute/path/to/model_ready
MODEL_PATH=/absolute/path/to/model.keras
```

The service calls the model with this categorical-plus-time input contract:

- `track_id`: string array
- `artist_id`: string array
- `album_name`: string array
- `hour_sin`: float array
- `hour_cos`: float array
- `day_sin`: float array
- `day_cos`: float array

If TensorFlow is not installed, the model file is missing, or prediction fails, the service falls back to a simple history-based score. That keeps the Pi service alive instead of crashing during music playback.

TensorFlow is not officially available for every Python version. Use Python 3.11 or 3.12 for training/runtime TensorFlow. The Windows venv that was available during scaffolding used Python 3.14, so tests can run there, but TensorFlow training should happen in a compatible environment.

The original Colab export is kept in `models/` as reference. Runtime training code lives in:

```text
src/rubik_spotify_queue/model_training.py
```

Runtime prediction code lives in `src/rubik_spotify_queue/recommender.py`.

## History Ingestion

Copy your Spotify Extended Streaming History JSON exports into:

```text
data/raw/
```

Then build model-ready data:

```bash
python -m rubik_spotify_queue.cli ingest-history
```

This writes:

- `data/model_ready/model_ready_history_full.csv`
- `data/model_ready/model_ready_history_train.csv`
- `data/model_ready/model_ready_history_val.csv`
- `data/model_ready/model_ready_history_test.csv`
- `data/model_ready/feature_config.json`
- `data/processed/history_events_processed_debug.csv`

Time features are continuous. `hour_sin/hour_cos` use fractional hour, and `day_sin/day_cos` use weekday plus fractional day.

## Local Files

The repo intentionally keeps personal data, secrets, and generated artifacts out of Git:

- `.env`: create it from `.env.example` and fill in Spotify app credentials plus local paths/settings.
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`: local Python environment and caches; recreate with the setup commands above.
- `data/spotify_tokens.json`: created by `python -m rubik_spotify_queue.cli login`; contains Spotify OAuth tokens.
- `data/*.duckdb` and `data/*.wal`: created by `python -m rubik_spotify_queue.cli init-db` and runtime polling/queueing.
- `data/raw/*`: your Spotify Extended Streaming History JSON exports; download them from Spotify and copy them into `data/raw/`.
- `data/model_ready/*` and `data/processed/*`: generated by `python -m rubik_spotify_queue.cli ingest-history`.
- `models/*.keras`, `models/*.h5`, `models/*.tflite`, `models/saved_model/`: generated by `python -m rubik_spotify_queue.cli train-model` or copied from another training machine.

The `.gitkeep` files under `data/` are committed only so the empty directory structure exists after cloning.

## History Seeding

Seed the candidate song table from model-ready history:

```bash
python -m rubik_spotify_queue.cli seed-history
```

This imports historical `track_id` / `artist_id` pairs from `model_ready_history_full.csv` into `songs`. The debug CSV provides the Spotify track URIs, so seeded songs are queueable immediately once Spotify login is complete.

## Runtime Modes

Polling and queueing can run separately:

```bash
python -m rubik_spotify_queue.cli poll
```

Records Spotify playback continuously using active/idle polling intervals. It ignores the queue window, so listening capture can run all day.

```bash
python -m rubik_spotify_queue.cli queue
```

Scores songs and adds tracks to your queue during the configured queue window. It does not record playback snapshots.
The queuer checks Spotify's visible queue with `/me/player/queue`. It maintains an app-managed buffer: when fewer than `QUEUE_TARGET_BUFFER_SIZE` app-queued tracks are still visible ahead, it fills every open slot up to that target. Candidate songs are also filtered by `CANDIDATE_MIN_TOTAL_PLAYS`, so one-off tracks from the raw export do not enter the recommendation pool by default.

When the queue is ready, selection works like this:

- Score queueable candidates with TensorFlow, or with the history fallback if TensorFlow is unavailable.
- Keep candidates above `MIN_CANDIDATE_TARGET_SCORE`.
- Sort by score and take the top `QUEUE_RANDOM_POOL_SIZE`.
- Randomly sample enough songs to fill the open `QUEUE_TARGET_BUFFER_SIZE` slots without replacement using `predicted_score ** QUEUE_SCORE_WEIGHT_POWER`.

```bash
python -m rubik_spotify_queue.cli serve
```

Runs the old combined mode in one process: polling plus queueing. Outside the queue window, active playback behaves normally; only no-playback checks use the long quiet-hours sleep.

For a no-write queueing test:

```bash
python -m rubik_spotify_queue.cli queue --dry-run
python -m rubik_spotify_queue.cli serve --dry-run
```

Dry run mode scores candidates and records `dry_run` queue actions, but it does not call Spotify's add-to-queue endpoint.

## Running safely on Rubik Pi

The long-running process is designed to survive multi-day runtime on a small board:

- TensorFlow/Keras is loaded once per service process and reused across queue checks.
- The in-memory playback snapshot buffer is bounded by `MAX_EVENT_BUFFER_SNAPSHOTS` (default `720`).
- App-queued track history is bounded by `QUEUE_TRACK_HISTORY_MAX` (default `200`).
- Unexpected loop failures use exponential backoff controlled by `INITIAL_ERROR_BACKOFF_SECONDS` and `MAX_ERROR_BACKOFF_SECONDS`.
- Every loop logs RSS memory, CPU percent, loop duration, sleep time, candidate counts, and buffer/cache sizes. Set `RESOURCE_LOG_EVERY_N_CYCLES` to reduce log volume.
- Optional Python allocation tracking is available with `MEMORY_DEBUG_TRACEMALLOC=true`.
- Raw Spotify API payloads are dropped before database insertion unless `PERSIST_RAW_SPOTIFY_PAYLOADS=true`.

Recommended Rubik Pi settings:

```text
RESOURCE_LOG_EVERY_N_CYCLES=1
MAX_EVENT_BUFFER_SNAPSHOTS=720
QUEUE_TRACK_HISTORY_MAX=200
PERSIST_RAW_SPOTIFY_PAYLOADS=false
MEMORY_DEBUG_TRACEMALLOC=false
```

Use systemd instead of tmux so the process gets a memory ceiling and clean restart behavior. A ready-to-edit service file is included at `deploy/rubik-spotify-queue.service`; change `User`, `Group`, `WorkingDirectory`, `EnvironmentFile`, and `ExecStart` to match your Pi paths.

```bash
sudo cp deploy/rubik-spotify-queue.service /etc/systemd/system/rubik-spotify-queue.service
sudo systemctl daemon-reload
sudo systemctl enable --now rubik-spotify-queue
journalctl -u rubik-spotify-queue -f
```

The example uses `Restart=always`, `RestartSec=10`, `MemoryMax=1500M`, and journal logging. If TensorFlow needs more room on your image, raise `MemoryMax` toward `2000M`; if the board starts swapping or heating badly, lower it toward `1200M` and run without TensorFlow.

## Reducing Rate Limits

Recommended starting values:

```text
ACTIVE_POLL_SECONDS=8
NEAR_TRACK_END_POLL_SECONDS=3
IDLE_POLL_SECONDS=120
QUIET_HOURS_POLL_SECONDS=900
QUEUE_START_HOUR=7
QUEUE_END_HOUR=24
QUEUE_READY_CHECK_SECONDS=20
QUEUE_ADD_COOLDOWN_SECONDS=10
QUEUE_TARGET_BUFFER_SIZE=2
QUEUE_RANDOM_POOL_SIZE=50
QUEUE_SCORE_WEIGHT_POWER=4.0
CANDIDATE_MIN_TOTAL_PLAYS=2
```

This avoids the old pattern of polling every second all day.

## Next Integration Step

Recommended first full run:

```bash
python -m rubik_spotify_queue.cli init-db
python -m rubik_spotify_queue.cli ingest-history
python -m rubik_spotify_queue.cli seed-history
python -m rubik_spotify_queue.cli train-model
python -m rubik_spotify_queue.cli health
python -m rubik_spotify_queue.cli login
python -m rubik_spotify_queue.cli poll
```

# Rubik Spotify Queue

Codex-written service for a Rubik Pi 3 that:

1. Polls Spotify playback politely and records finalized listening events.
2. Scores known songs with a TensorFlow model when available.
3. Adds the highest-scoring song to your Spotify queue when it is likely not to be skipped.

The old `music-ai-recommender` project is kept as reference. This project is the cleaner service target.

## Design

- `snapshots` stores raw currently-playing observations.
- `listening_events` stores one row per finished track.
- `songs` stores the local candidate universe.
- `queue_actions` records every queue attempt.

Polling is adaptive:

- `ACTIVE_POLL_SECONDS`: normal playback polling.
- `NEAR_TRACK_END_POLL_SECONDS`: faster checks near the end of a song.
- `IDLE_POLL_SECONDS`: slower checks when nothing is playing.
- `QUIET_HOURS_POLL_SECONDS`: very slow checks outside the configured queue window.
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
rubik-spotify-queue seed-history
rubik-spotify-queue login
rubik-spotify-queue serve
```

On a Rubik Pi/Linux shell:

```bash
cd rubik-spotify-queue
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m rubik_spotify_queue.cli init-db
python -m rubik_spotify_queue.cli seed-history
python -m rubik_spotify_queue.cli serve
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
../music-ai-recommender/data/model_ready/model_ready_history_train.csv
../music-ai-recommender/data/model_ready/model_ready_history_val.csv
../music-ai-recommender/data/model_ready/model_ready_history_test.csv
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

The service calls the model with the same six-input contract used by the Colab export:

- `track_id`: string array
- `artist_id`: string array
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

## History Seeding

Seed the candidate song table from model-ready history:

```bash
python -m rubik_spotify_queue.cli seed-history
```

This imports unique `track_id` / `artist_id` pairs from `model_ready_history_full.csv` into `songs`. Tracks with `spotify:track:...` IDs are queueable immediately once Spotify login is complete.

## Reducing Rate Limits

Recommended starting values:

```text
ACTIVE_POLL_SECONDS=8
NEAR_TRACK_END_POLL_SECONDS=3
IDLE_POLL_SECONDS=120
QUIET_HOURS_POLL_SECONDS=900
QUEUE_START_HOUR=7
QUEUE_END_HOUR=24
MIN_QUEUE_INTERVAL_SECONDS=180
```

This avoids the old pattern of polling every second all day.

## systemd Example

Create `/etc/systemd/system/rubik-spotify-queue.service`:

```ini
[Unit]
Description=Rubik Spotify Queue
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/pi/rubik-spotify-queue
ExecStart=/home/pi/rubik-spotify-queue/.venv/bin/python -m rubik_spotify_queue.cli serve
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rubik-spotify-queue
journalctl -u rubik-spotify-queue -f
```

## Next Integration Step

Recommended first full run:

```bash
python -m rubik_spotify_queue.cli init-db
python -m rubik_spotify_queue.cli seed-history
python -m rubik_spotify_queue.cli train-model
python -m rubik_spotify_queue.cli health
python -m rubik_spotify_queue.cli login
python -m rubik_spotify_queue.cli serve
```

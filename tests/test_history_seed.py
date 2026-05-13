from __future__ import annotations

import duckdb
import pandas as pd

from rubik_spotify_queue.db import migrate
from rubik_spotify_queue.history_seed import seed_songs_from_model_ready


def test_seed_songs_from_model_ready_populates_songs(tmp_path) -> None:
    model_ready = tmp_path / "model_ready"
    model_ready.mkdir()
    pd.DataFrame(
        {
            "track_id": ["One", "One", "Two"],
            "artist_id": ["Artist 1", "Artist 1", "Artist 2"],
            "target_score": [1.0, 0.8, 0.4],
        }
    ).to_csv(model_ready / "model_ready_history_full.csv", index=False)
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "track_id": ["spotify:track:one", "spotify:track:two"],
            "track_name": ["One", "Two"],
            "artist_id": ["Artist 1", "Artist 2"],
        }
    ).to_csv(processed / "history_events_processed_debug.csv", index=False)

    db_path = tmp_path / "service.duckdb"
    migrate(db_path)
    con = duckdb.connect(str(db_path))
    try:
        result = seed_songs_from_model_ready(con, model_ready)
        count = con.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        uri = con.execute("SELECT spotify_uri FROM songs WHERE track_id = 'One'").fetchone()[0]
    finally:
        con.close()

    assert result.rows_read == 3
    assert result.songs_upserted == 2
    assert result.queueable_songs == 2
    assert count == 2
    assert uri == "spotify:track:one"

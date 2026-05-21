"""Train the Colab-derived TensorFlow recommender as a saved Keras model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TRACK_COLUMN = "track_id"
ARTIST_COLUMN = "artist_id"
ALBUM_COLUMN = "album_name"
NUMERIC_COLUMNS = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
MODEL_FEATURES = [TRACK_COLUMN, ARTIST_COLUMN, ALBUM_COLUMN, *NUMERIC_COLUMNS]
TARGET_COLUMN = "target_score"


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    train_rows: int
    val_rows: int
    test_rows: int
    test_loss: float
    test_mae: float


def load_model_ready_frames(model_ready_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(model_ready_dir / "model_ready_history_train.csv")
    val = pd.read_csv(model_ready_dir / "model_ready_history_val.csv")
    test = pd.read_csv(model_ready_dir / "model_ready_history_test.csv")
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        missing = [col for col in [*MODEL_FEATURES, TARGET_COLUMN] if col not in frame.columns]
        if missing:
            raise ValueError(f"{name} CSV is missing required columns: {missing}")
    return train, val, test


def input_dict(frame: pd.DataFrame) -> dict[str, Any]:
    import tensorflow as tf  # type: ignore[import-not-found]

    return {
        "track_id": tf.constant(frame[TRACK_COLUMN].astype(str).to_numpy(), dtype=tf.string),
        "artist_id": tf.constant(frame[ARTIST_COLUMN].astype(str).to_numpy(), dtype=tf.string),
        "album_name": tf.constant(frame[ALBUM_COLUMN].fillna("unknown").astype(str).to_numpy(), dtype=tf.string),
        "hour_sin": tf.constant(frame["hour_sin"].astype("float32").to_numpy(), dtype=tf.float32),
        "hour_cos": tf.constant(frame["hour_cos"].astype("float32").to_numpy(), dtype=tf.float32),
        "day_sin": tf.constant(frame["day_sin"].astype("float32").to_numpy(), dtype=tf.float32),
        "day_cos": tf.constant(frame["day_cos"].astype("float32").to_numpy(), dtype=tf.float32),
    }


def build_model(train_df: pd.DataFrame, *, seed: int = 552026):
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
    from tensorflow.keras import layers, models  # type: ignore[import-not-found]

    keras.utils.set_random_seed(seed)

    track_lookup = layers.StringLookup(
        vocabulary=train_df[TRACK_COLUMN].astype(str).unique(),
        mask_token=None,
        num_oov_indices=1,
    )
    artist_lookup = layers.StringLookup(
        vocabulary=train_df[ARTIST_COLUMN].astype(str).unique(),
        mask_token=None,
        num_oov_indices=1,
    )
    album_lookup = layers.StringLookup(
        vocabulary=train_df[ALBUM_COLUMN].fillna("unknown").astype(str).unique(),
        mask_token=None,
        num_oov_indices=1,
    )

    track_input = keras.Input(shape=(1,), name="track_id", dtype=tf.string)
    artist_input = keras.Input(shape=(1,), name="artist_id", dtype=tf.string)
    album_input = keras.Input(shape=(1,), name="album_name", dtype=tf.string)
    hour_sin_input = keras.Input(shape=(1,), name="hour_sin", dtype=tf.float32)
    hour_cos_input = keras.Input(shape=(1,), name="hour_cos", dtype=tf.float32)
    day_sin_input = keras.Input(shape=(1,), name="day_sin", dtype=tf.float32)
    day_cos_input = keras.Input(shape=(1,), name="day_cos", dtype=tf.float32)

    track_vector = layers.Flatten()(layers.Embedding(track_lookup.vocabulary_size(), 32)(track_lookup(track_input)))
    artist_vector = layers.Flatten()(layers.Embedding(artist_lookup.vocabulary_size(), 16)(artist_lookup(artist_input)))
    album_vector = layers.Flatten()(layers.Embedding(album_lookup.vocabulary_size(), 16)(album_lookup(album_input)))

    full_input_layer = layers.Concatenate()(
        [
            track_vector,
            artist_vector,
            album_vector,
            hour_sin_input,
            hour_cos_input,
            day_sin_input,
            day_cos_input,
        ]
    )

    prediction_head = models.Sequential(
        [
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model = models.Model(
        inputs=[
            track_input,
            artist_input,
            album_input,
            hour_sin_input,
            hour_cos_input,
            day_sin_input,
            day_cos_input,
        ],
        outputs=prediction_head(full_input_layer),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_and_save(
    *,
    model_ready_dir: Path,
    model_path: Path,
    epochs: int = 20,
    batch_size: int = 32,
) -> TrainingResult:
    train_df, val_df, test_df = load_model_ready_frames(model_ready_dir)
    model = build_model(train_df)

    model.fit(
        input_dict(train_df),
        train_df[TARGET_COLUMN].astype("float32").to_numpy(),
        validation_data=(input_dict(val_df), val_df[TARGET_COLUMN].astype("float32").to_numpy()),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    test_loss, test_mae = model.evaluate(
        input_dict(test_df),
        test_df[TARGET_COLUMN].astype("float32").to_numpy(),
        verbose=0,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    return TrainingResult(
        model_path=model_path,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        test_loss=float(test_loss),
        test_mae=float(test_mae),
    )

"""Shared configuration constants for the Music Mood Predictor.

This module is the single source of truth for the feature set, target column,
mood classes, per-feature validation ranges, reproducibility settings, and
filesystem paths used across the data pipeline and the Streamlit web app.
"""

from __future__ import annotations

# Canonical ordering of the eight numeric audio features. This order is used
# everywhere features are consumed (preprocessing, training, prediction) so the
# model always receives inputs in a consistent layout.
FEATURES: list[str] = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "acousticness",
    "speechiness",
    "instrumentalness",
]

# Name of the ground-truth label column in the dataset.
TARGET: str = "mood"

# The four mood classes the system predicts.
MOOD_CLASSES: list[str] = ["Happy", "Sad", "Calm", "Energetic"]

# Valid (min, max) range for each audio feature. Used by both preprocessing
# validation and the web app input validation. Tempo (BPM) and loudness (dB)
# are NOT on a 0-1 scale; ranges are slightly wider than the observed data to
# tolerate valid real-world inputs while rejecting nonsense values.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (0.0, 250.0),
    "loudness": (-60.0, 5.0),
    "acousticness": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
}

# Fixed random seed for reproducible splits and model training.
RANDOM_SEED: int = 42

# Fraction of the dataset held out for the test set.
TEST_SIZE: float = 0.2

# Filesystem paths (relative to the project root).
DATA_PATH: str = "data/spotify_labeled.csv"
MODEL_PATH: str = "models/music_mood_model.pkl"

# Short, user-facing textual description for each mood class, surfaced on the
# prediction page after a mood is predicted (Requirement 9.5).
MOOD_DESCRIPTIONS: dict[str, str] = {
    "Happy": (
        "Upbeat and positive - bright, cheerful tracks with high valence that "
        "lift your spirits and make you want to sing along."
    ),
    "Sad": (
        "Melancholic and reflective - low-energy, low-valence songs that carry "
        "an emotional, wistful mood."
    ),
    "Calm": (
        "Relaxed and soothing - mellow, often acoustic tracks with gentle "
        "energy, ideal for unwinding or focus."
    ),
    "Energetic": (
        "High-energy and driving - intense, danceable tracks with strong tempo "
        "and power that get you moving."
    ),
}

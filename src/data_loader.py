"""Dataset ingestion and column validation for the Music Mood Predictor.

Provides :func:`load_dataset`, which reads the labeled Spotify CSV from the
workspace and validates that all eight audio features plus the target column
are present. Missing files and missing columns raise descriptive errors that
name the offending path or column(s) (Requirements 2.2, 2.3, 2.4, 12.2).
"""

from __future__ import annotations

import os

import pandas as pd

from src.config import DATA_PATH, FEATURES, TARGET
from src.logger import get_logger

logger = get_logger(__name__)


class MissingColumnError(Exception):
    """Raised when the dataset is missing one or more required columns.

    The error message names the missing column(s) so callers can identify
    exactly which of the eight audio features or the target column is absent
    (Requirement 2.4).
    """


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the labeled Spotify dataset and validate its schema.

    Reads the CSV at ``path`` into a :class:`pandas.DataFrame` and verifies that
    all eight audio features (``FEATURES``) and the target column (``TARGET``)
    are present.

    Args:
        path: Filesystem path to the dataset CSV. Defaults to ``DATA_PATH``.

    Returns:
        A DataFrame containing at least the eight audio features and the target
        column.

    Raises:
        FileNotFoundError: If no file exists at ``path``. The message names the
            missing path (Requirement 2.3).
        MissingColumnError: If any required column is absent from the loaded
            data. The message names the missing column(s) (Requirement 2.4).
    """
    if not os.path.isfile(path):
        logger.error("Dataset file not found at path: %s", path)
        raise FileNotFoundError(
            f"Dataset file not found at expected path: '{path}'. "
            "Ensure 'spotify_labeled.csv' is present in the workspace."
        )

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    required_columns = [*FEATURES, TARGET]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        logger.error("Dataset is missing required column(s): %s", missing_str)
        raise MissingColumnError(
            f"Dataset is missing required column(s): {missing_str}. "
            f"Expected the 8 audio features and the target column '{TARGET}'."
        )

    logger.info("Loaded dataset with %d rows and %d columns", len(df), df.shape[1])
    return df

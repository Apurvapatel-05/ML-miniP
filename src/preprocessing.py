"""Data preprocessing for the Music Mood Predictor.

Cleans the raw dataset and produces model-ready, validated train/test splits.
Steps (Requirements 4.1-4.6, 12.2):

1. Remove duplicate rows.
2. Drop rows with a missing target value.
3. Label-encode the ``mood`` target into integers.
4. Select exactly the eight audio features.
5. Median-impute missing feature values using the training-set median (the
   median is learned on the training split only to avoid leakage).
6. Stratified train/test split with a fixed random seed.
7. Post-split validation: feature frames contain exactly the eight numeric
   features with no missing values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURES, RANDOM_SEED, TARGET, TEST_SIZE
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Model-ready, validated train/test split.

    Attributes:
        X_train: Training feature frame (exactly the 8 numeric features).
        X_test: Test feature frame (exactly the 8 numeric features).
        y_train: Integer-encoded training labels.
        y_test: Integer-encoded test labels.
        label_encoder: The fitted :class:`LabelEncoder` for the target.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


def encode_labels(moods: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Label-encode mood strings into integers.

    Args:
        moods: A series of mood class labels.

    Returns:
        A tuple of the integer-encoded label array and the fitted
        :class:`LabelEncoder` (usable for inverse transforms).
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(moods)
    return encoded, encoder


def validate_features(X: pd.DataFrame) -> None:
    """Validate that a feature frame is model-ready.

    Ensures the frame contains exactly the eight audio features (in order),
    that every column is numeric, and that there are no missing values.

    Args:
        X: A candidate feature frame.

    Raises:
        ValueError: If the columns do not match the expected eight features,
            if any column is non-numeric, or if any value is missing.
    """
    actual = list(X.columns)
    if actual != FEATURES:
        raise ValueError(
            f"Feature frame must contain exactly the 8 audio features "
            f"{FEATURES}, but got {actual}."
        )

    non_numeric = [col for col in X.columns if not is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(
            f"Feature columns must be numeric; non-numeric columns: {non_numeric}."
        )

    missing_cols = [col for col in X.columns if X[col].isna().any()]
    if missing_cols:
        raise ValueError(
            f"Feature frame contains missing values in columns: {missing_cols}."
        )


def preprocess(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> SplitData:
    """Clean, encode, split, and validate the dataset.

    Args:
        df: Raw dataset containing the eight audio features and the target.
        test_size: Fraction of rows to allocate to the test set.
        seed: Random seed for the reproducible stratified split.

    Returns:
        A :class:`SplitData` with validated, model-ready train/test frames and
        the fitted label encoder.

    Raises:
        ValueError: If, after cleaning, there is insufficient data to perform a
            stratified split, or if post-split validation fails.
    """
    logger.info("Preprocessing dataset with %d raw rows", len(df))

    # 1. Remove duplicate rows.
    before = len(df)
    work = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate rows", before - len(work))

    # 2. Drop rows with a missing target value.
    before = len(work)
    work = work.dropna(subset=[TARGET]).reset_index(drop=True)
    logger.info("Dropped %d rows with a missing target", before - len(work))

    if work.empty:
        raise ValueError("No rows remain after removing duplicates and missing targets.")

    # 3. Label-encode the target.
    y_all, label_encoder = encode_labels(work[TARGET])

    # 4. Select exactly the eight audio features (canonical order).
    X_all = work[FEATURES].copy()

    # 6. Stratified train/test split with a fixed seed (before imputation so the
    #    imputation median is learned on the training split only).
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all,
    )

    # 5. Median-impute missing feature values using the training-set median.
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    # Guard against a feature that is entirely NaN in the training split, which
    # would leave a NaN median; fall back to 0.0 for such columns.
    if X_train.isna().any().any() or X_test.isna().any().any():
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)

    # Ensure numeric dtype for all feature columns.
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # 7. Post-split validation.
    validate_features(X_train)
    validate_features(X_test)

    logger.info(
        "Preprocessing complete: %d train rows, %d test rows",
        len(X_train),
        len(X_test),
    )

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
    )

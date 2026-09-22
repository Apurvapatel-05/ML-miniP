"""Model training for the Music Mood Predictor.

Trains a Random Forest Classifier (the only algorithm used) with hyperparameter
tuning via ``GridSearchCV`` and k-fold cross-validation via ``StratifiedKFold``.
Training uses a fixed random seed for reproducibility (Requirements 5.1-5.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.config import RANDOM_SEED
from src.logger import get_logger

logger = get_logger(__name__)

# Hyperparameter search space for the Random Forest (Requirement 5.2).
PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}

# Number of cross-validation folds (Requirement 5.3).
CV_FOLDS: int = 5


@dataclass
class TrainResult:
    """Result of training the Classifier.

    Attributes:
        model: The fitted :class:`RandomForestClassifier` (best estimator).
        best_params: The best hyperparameters found by the grid search.
        cv_mean: Mean cross-validation accuracy of the tuned model.
        cv_std: Standard deviation of the cross-validation accuracy.
    """

    model: RandomForestClassifier
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = RANDOM_SEED,
    param_grid: dict[str, list[Any]] | None = None,
) -> TrainResult:
    """Train and tune a Random Forest Classifier.

    Args:
        X_train: Training feature frame (the eight numeric audio features).
        y_train: Integer-encoded training labels.
        seed: Random seed for reproducible training.
        param_grid: Optional override of the hyperparameter search space
            (used to keep tests fast).

    Returns:
        A :class:`TrainResult` with the fitted best model, best parameters, and
        cross-validation accuracy mean/std.
    """
    grid = param_grid if param_grid is not None else PARAM_GRID

    # Number of folds is bounded by the smallest class count so CV never fails
    # on tiny datasets (used in tests); at least 2 folds.
    _, class_counts = np.unique(y_train, return_counts=True)
    n_splits = max(2, min(CV_FOLDS, int(class_counts.min())))

    logger.info(
        "Training Random Forest with GridSearchCV over %d param combos, %d-fold CV",
        _count_combos(grid),
        n_splits,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    base = RandomForestClassifier(class_weight="balanced", random_state=seed)
    search = GridSearchCV(
        estimator=base,
        param_grid=grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    # Report mean/std CV accuracy of the tuned model (Requirement 5.3).
    scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_mean = float(scores.mean())
    cv_std = float(scores.std())

    logger.info(
        "Training complete. Best params: %s | CV accuracy: %.4f +/- %.4f",
        best_params,
        cv_mean,
        cv_std,
    )

    return TrainResult(
        model=best_model,
        best_params=best_params,
        cv_mean=cv_mean,
        cv_std=cv_std,
    )


def _count_combos(grid: dict[str, list[Any]]) -> int:
    """Return the number of hyperparameter combinations in a grid."""
    total = 1
    for values in grid.values():
        total *= len(values)
    return total

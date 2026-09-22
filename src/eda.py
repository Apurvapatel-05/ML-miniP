"""Exploratory Data Analysis for the Music Mood Predictor.

Provides :func:`run_eda`, which computes descriptive statistics about the
dataset and saves visualization plots (mood class distribution, audio feature
distributions, feature correlation matrix) as PNG files. A non-interactive
matplotlib backend (Agg) is selected so the module works in headless
environments (Requirements 3.1-3.7).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import matplotlib

# Use a non-interactive backend so plot generation works headless (no display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import FEATURES, MOOD_CLASSES, TARGET  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class EdaReport:
    """Summary statistics produced by the EDA module.

    Attributes:
        row_count: Number of rows in the dataset.
        col_count: Number of columns in the dataset.
        dtypes: Mapping of column name to its data type (as a string).
        missing_counts: Mapping of column name to count of missing values.
        duplicate_count: Number of fully duplicated rows.
        class_distribution: Mapping of mood class to its row count.
        feature_summary: Per-feature summary statistics (describe output).
        class_imbalance_note: Human-readable note about class imbalance.
        plot_paths: Mapping of plot name to the saved PNG file path.
    """

    row_count: int
    col_count: int
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_count: int
    class_distribution: dict[str, int]
    feature_summary: dict[str, dict[str, float]]
    class_imbalance_note: str = ""
    plot_paths: dict[str, str] = field(default_factory=dict)


def _plot_class_distribution(df: pd.DataFrame, out_dir: str) -> str:
    """Save a bar chart of the mood class distribution. Returns the file path."""
    path = os.path.join(out_dir, "class_distribution.png")
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False)
    ax.set_title("Mood Class Distribution")
    ax.set_xlabel("Mood")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _plot_feature_distributions(df: pd.DataFrame, out_dir: str) -> str:
    """Save histograms for each audio feature. Returns the file path."""
    path = os.path.join(out_dir, "feature_distributions.png")
    n = len(FEATURES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        sns.histplot(df[feature].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(feature)
    # Hide any unused subplot axes.
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Audio Feature Distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _plot_correlation_matrix(df: pd.DataFrame, out_dir: str) -> str:
    """Save a heatmap of the feature correlation matrix. Returns the file path."""
    path = os.path.join(out_dir, "correlation_matrix.png")
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma", ax=ax)
    ax.set_title("Audio Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _build_imbalance_note(class_distribution: dict[str, int]) -> str:
    """Return a human-readable note describing class imbalance, if any."""
    if not class_distribution:
        return "No class distribution available."
    counts = list(class_distribution.values())
    max_count = max(counts)
    min_count = min(counts)
    ratio = max_count / min_count if min_count > 0 else float("inf")
    most = max(class_distribution, key=class_distribution.get)
    least = min(class_distribution, key=class_distribution.get)
    if ratio >= 1.5:
        return (
            f"Class imbalance detected: '{most}' ({max_count}) is the most common "
            f"class and '{least}' ({min_count}) the least common "
            f"(ratio {ratio:.2f}:1). Use stratified splitting and a "
            "class-weight-balanced classifier."
        )
    return (
        f"Classes are reasonably balanced (max/min ratio {ratio:.2f}:1); "
        "stratified splitting is still recommended."
    )


def run_eda(df: pd.DataFrame, out_dir: str = "outputs/eda") -> EdaReport:
    """Compute dataset statistics and save EDA visualization plots.

    Args:
        df: The loaded dataset containing the eight audio features and target.
        out_dir: Directory where PNG plots are saved. Created if it does not
            exist.

    Returns:
        An :class:`EdaReport` populated with dataset statistics, a class
        imbalance note, and the paths to the saved plot images.
    """
    logger.info("Running EDA on dataset with %d rows", len(df))
    os.makedirs(out_dir, exist_ok=True)

    row_count = int(df.shape[0])
    col_count = int(df.shape[1])
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = {col: int(count) for col, count in df.isna().sum().items()}
    duplicate_count = int(df.duplicated().sum())

    # Class distribution ordered by the canonical mood class list where present.
    raw_counts: dict[Any, int] = df[TARGET].value_counts().to_dict()
    class_distribution = {
        cls: int(raw_counts.get(cls, 0)) for cls in MOOD_CLASSES if cls in raw_counts
    }
    # Include any unexpected classes that appear in the data for completeness.
    for cls, cnt in raw_counts.items():
        if cls not in class_distribution:
            class_distribution[str(cls)] = int(cnt)

    # Per-feature summary statistics (describe output as nested dict).
    feature_summary = {
        feature: {
            stat: float(value)
            for stat, value in df[feature].describe().to_dict().items()
        }
        for feature in FEATURES
    }

    imbalance_note = _build_imbalance_note(class_distribution)
    logger.info(imbalance_note)

    plot_paths = {
        "class_distribution": _plot_class_distribution(df, out_dir),
        "feature_distributions": _plot_feature_distributions(df, out_dir),
        "correlation_matrix": _plot_correlation_matrix(df, out_dir),
    }
    logger.info("Saved %d EDA plots to %s", len(plot_paths), out_dir)

    return EdaReport(
        row_count=row_count,
        col_count=col_count,
        dtypes=dtypes,
        missing_counts=missing_counts,
        duplicate_count=duplicate_count,
        class_distribution=class_distribution,
        feature_summary=feature_summary,
        class_imbalance_note=imbalance_note,
        plot_paths=plot_paths,
    )

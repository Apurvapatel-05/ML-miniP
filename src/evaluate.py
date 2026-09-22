"""Model evaluation for the Music Mood Predictor.

Computes classification metrics (accuracy, precision, recall, F1), a confusion
matrix, a per-class classification report, and feature importances, and saves
the confusion matrix and feature importance visualizations as PNG files
(Requirements 6.1-6.5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FEATURES  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class EvalReport:
    """Evaluation metrics and artifact paths.

    Attributes:
        accuracy: Overall test accuracy.
        precision: Macro-averaged precision.
        recall: Macro-averaged recall.
        f1: Macro-averaged F1 score.
        confusion: Confusion matrix as a nested list.
        classification_report: Per-class report dict.
        feature_importance: Mapping of feature name to importance.
        plot_paths: Mapping of plot name to saved PNG path.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]
    classification_report: dict[str, Any]
    feature_importance: dict[str, float]
    plot_paths: dict[str, str] = field(default_factory=dict)


def _plot_confusion(cm: np.ndarray, class_names: list[str], out_dir: str) -> str:
    """Save the confusion matrix heatmap. Returns the file path."""
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted mood")
    ax.set_ylabel("True mood")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _plot_feature_importance(importance: dict[str, float], out_dir: str) -> str:
    """Save the feature importance bar chart. Returns the file path."""
    path = os.path.join(out_dir, "feature_importance.png")
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=values, y=names, ax=ax, hue=names, legend=False, palette="magma")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Audio feature")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder,
    out_dir: str = "outputs/evaluation",
) -> EvalReport:
    """Evaluate a trained model on the test set and save plots.

    Args:
        model: The fitted classifier.
        X_test: Test feature frame.
        y_test: Integer-encoded true labels.
        label_encoder: Fitted LabelEncoder mapping integers to mood names.
        out_dir: Directory where evaluation plots are saved.

    Returns:
        A populated :class:`EvalReport`.
    """
    logger.info("Evaluating model on %d test rows", len(X_test))
    os.makedirs(out_dir, exist_ok=True)

    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    class_names = [str(c) for c in label_encoder.classes_]
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        feature_importance = {
            feat: float(imp) for feat, imp in zip(FEATURES, importances)
        }
    else:
        feature_importance = {feat: 0.0 for feat in FEATURES}

    plot_paths = {
        "confusion_matrix": _plot_confusion(cm, class_names, out_dir),
        "feature_importance": _plot_feature_importance(feature_importance, out_dir),
    }

    logger.info(
        "Evaluation complete. accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        accuracy,
        precision,
        recall,
        f1,
    )

    return EvalReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=cm.tolist(),
        classification_report=report,
        feature_importance=feature_importance,
        plot_paths=plot_paths,
    )

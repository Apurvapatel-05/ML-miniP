"""Model persistence and prediction for the Music Mood Predictor.

Saves and loads a model bundle ``{model, label_encoder, feature_order,
metadata}`` via joblib to ``music_mood_model.pkl`` and provides a
``ModelBundle`` that turns a dict of audio features into a ``Prediction``
(mood + confidence + probabilities). Requirements 7.1-7.4, 1.5, 9.3, 9.4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES, MODEL_PATH
from src.logger import get_logger

logger = get_logger(__name__)


class ModelFileNotFoundError(FileNotFoundError):
    """Raised when the serialized model file cannot be found (Requirement 7.4)."""


@dataclass
class Prediction:
    """A single mood prediction.

    Attributes:
        mood: The predicted mood class (one of the four classes).
        confidence: The predicted probability for ``mood``, in [0, 100].
        probabilities: Mapping of every mood class to its probability (%).
    """

    mood: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class ModelBundle:
    """A loaded model plus everything needed to make predictions.

    Attributes:
        model: The fitted classifier.
        label_encoder: Fitted LabelEncoder (int <-> mood name).
        feature_order: Canonical order of the eight features.
        metadata: Training/evaluation metadata.
    """

    model: Any
    label_encoder: Any
    feature_order: list[str] = field(default_factory=lambda: list(FEATURES))
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, features: dict[str, float]) -> Prediction:
        """Predict the mood for a dict of audio features.

        Args:
            features: Mapping of each of the eight features to a numeric value.

        Returns:
            A :class:`Prediction` with the mood, confidence (0-100), and the
            full class-probability map.
        """
        row = pd.DataFrame([[float(features[f]) for f in self.feature_order]],
                           columns=self.feature_order)
        proba = self.model.predict_proba(row)[0]
        class_indices = self.model.classes_
        mood_names = self.label_encoder.inverse_transform(class_indices)

        probabilities = {
            str(name): round(float(p) * 100.0, 2)
            for name, p in zip(mood_names, proba)
        }
        best_idx = int(np.argmax(proba))
        mood = str(mood_names[best_idx])
        confidence = round(float(proba[best_idx]) * 100.0, 2)

        return Prediction(mood=mood, confidence=confidence, probabilities=probabilities)


def save_model(
    model,
    label_encoder,
    metadata: dict[str, Any] | None = None,
    path: str = MODEL_PATH,
) -> None:
    """Serialize the model bundle to ``path`` via joblib (Requirement 7.1).

    Args:
        model: The fitted classifier.
        label_encoder: Fitted LabelEncoder.
        metadata: Optional training/evaluation metadata.
        path: Destination path for the ``.pkl`` file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_order": list(FEATURES),
        "metadata": metadata or {},
    }
    joblib.dump(bundle, path)
    logger.info("Saved model bundle to %s", path)


def load_model(path: str = MODEL_PATH) -> ModelBundle:
    """Load the model bundle from ``path`` (Requirements 7.2, 7.4).

    Args:
        path: Path to the serialized ``.pkl`` file.

    Returns:
        A :class:`ModelBundle`.

    Raises:
        ModelFileNotFoundError: If the file does not exist, with a descriptive
            message telling the user to run the training pipeline.
    """
    if not os.path.isfile(path):
        logger.error("Model file not found at %s", path)
        raise ModelFileNotFoundError(
            f"Model file '{path}' not found. Run the training pipeline "
            "(`python run_pipeline.py`) to generate 'music_mood_model.pkl'."
        )

    bundle = joblib.load(path)
    logger.info("Loaded model bundle from %s", path)
    return ModelBundle(
        model=bundle["model"],
        label_encoder=bundle["label_encoder"],
        feature_order=bundle.get("feature_order", list(FEATURES)),
        metadata=bundle.get("metadata", {}),
    )

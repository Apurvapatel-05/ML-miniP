"""Shared input validation for the Music Mood Predictor.

Validates that a set of submitted audio-feature values are numeric and within
their defined valid ranges (``FEATURE_RANGES``) before a prediction is made.
Used by both the Streamlit prediction page and any programmatic caller
(Requirements 9.7, 12.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import FEATURE_RANGES, FEATURES
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationError:
    """A single per-field validation failure.

    Attributes:
        feature: The name of the offending audio feature.
        message: A human-readable description of why the value is invalid.
    """

    feature: str
    message: str


def validate_input(values: dict[str, float]) -> list[ValidationError]:
    """Validate submitted audio-feature values.

    Each of the eight audio features must be present, numeric (not NaN/inf), and
    within its ``FEATURE_RANGES`` bounds.

    Args:
        values: Mapping of feature name to submitted value.

    Returns:
        A list of :class:`ValidationError`. An empty list means the input is
        valid and a prediction may proceed.
    """
    errors: list[ValidationError] = []

    for feature in FEATURES:
        low, high = FEATURE_RANGES[feature]

        if feature not in values:
            errors.append(ValidationError(feature, f"'{feature}' is required."))
            continue

        value = values[feature]

        # Numeric check (reject bools, strings, None, NaN, inf).
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(
                ValidationError(feature, f"'{feature}' must be a number.")
            )
            continue

        if math.isnan(value) or math.isinf(value):
            errors.append(
                ValidationError(feature, f"'{feature}' must be a finite number.")
            )
            continue

        # Range check.
        if value < low or value > high:
            errors.append(
                ValidationError(
                    feature,
                    f"'{feature}' must be between {low} and {high} "
                    f"(got {value}).",
                )
            )

    if errors:
        logger.info("Input validation found %d error(s)", len(errors))

    return errors

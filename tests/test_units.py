"""Example / edge-case unit tests for the Music Mood Predictor.

These complement the property-based tests in ``tests/test_properties.py`` by
covering specific behaviors and formats.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re

from src.logger import LOG_FORMAT, get_logger


def _format_record(logger: logging.Logger, level: int, message: str) -> str:
    """Format a log record the same way the logger's handlers do.

    Returns the rendered log line so the test can assert on its contents.
    """
    formatter = logging.Formatter(LOG_FORMAT)
    record = logger.makeRecord(
        logger.name, level, fn="test", lno=0, msg=message, args=(), exc_info=None
    )
    return formatter.format(record)


def test_logger_record_carries_timestamp_and_severity() -> None:
    """Emitted log records include a timestamp and a severity level.

    Validates Requirement 12.3: the Logger SHALL record events with a
    timestamp and severity level.
    """
    logger = get_logger("test.logger.format")

    line = _format_record(logger, logging.WARNING, "something happened")

    # Severity level name is present in the record.
    assert "WARNING" in line

    # The logger name and message are present.
    assert "test.logger.format" in line
    assert "something happened" in line

    # A parseable timestamp is present at the start of the line
    # (default asctime format: "YYYY-MM-DD HH:MM:SS,mmm").
    timestamp_part = line.split(" | ", 1)[0]
    match = re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", timestamp_part)
    assert match is not None, f"no timestamp found in log line: {line!r}"

    # The timestamp is valid and parseable as a datetime.
    parsed = _dt.datetime.strptime(match.group(0), "%Y-%m-%d %H:%M:%S,%f")
    assert isinstance(parsed, _dt.datetime)


def test_get_logger_does_not_add_duplicate_handlers() -> None:
    """Repeated calls for the same logger name reuse handlers.

    Guards against duplicate log output when ``get_logger`` is called multiple
    times for the same module (supports Requirement 12.2/12.3 reliability).
    """
    logger_a = get_logger("test.logger.dedup")
    handler_count_first = len(logger_a.handlers)

    logger_b = get_logger("test.logger.dedup")
    handler_count_second = len(logger_b.handlers)

    assert logger_a is logger_b
    assert handler_count_first == handler_count_second
    assert handler_count_first >= 1


# ---------------------------------------------------------------------------
# Data loader unit tests (Task 3.3)
# ---------------------------------------------------------------------------

import os

import pandas as pd
import pytest

from src.config import DATA_PATH, FEATURES, TARGET
from src.data_loader import MissingColumnError, load_dataset


def test_load_dataset_loads_expected_columns() -> None:
    """The loaded dataset contains the 8 audio features and target column.

    Validates Requirement 2.2: the pipeline reads the eight audio features and
    the target column into a data structure.
    """
    # Only run against the real dataset when it is present in the workspace.
    if not os.path.isfile(DATA_PATH):
        pytest.skip(f"dataset not present at {DATA_PATH}")

    df = load_dataset()

    for column in [*FEATURES, TARGET]:
        assert column in df.columns
    assert len(df) > 0


def test_load_dataset_missing_file_error_names_path() -> None:
    """A missing dataset file raises FileNotFoundError naming the path.

    Validates Requirement 2.3: a descriptive error identifies the missing file.
    """
    missing_path = "data/does_not_exist_spotify.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(missing_path)

    assert missing_path in str(exc_info.value)


def test_load_dataset_missing_column_error_names_column(tmp_path) -> None:
    """A dataset missing a required column raises MissingColumnError naming it.

    Validates Requirement 2.4.
    """
    data = {col: [0.5, 0.6] for col in FEATURES if col != "tempo"}
    data[TARGET] = ["Happy", "Sad"]
    csv_path = tmp_path / "no_tempo.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    with pytest.raises(MissingColumnError) as exc_info:
        load_dataset(str(csv_path))

    assert "tempo" in str(exc_info.value)


# ---------------------------------------------------------------------------
# EDA unit tests (Task 4.2)
# ---------------------------------------------------------------------------

from src.config import MOOD_CLASSES
from src.eda import EdaReport, run_eda


def _sample_dataframe() -> pd.DataFrame:
    """Build a small in-memory dataset covering all four mood classes."""
    rows = []
    for i, mood in enumerate(MOOD_CLASSES * 3):
        row = {feature: 0.1 * (i + 1) % 1.0 for feature in FEATURES}
        row["tempo"] = 100.0 + i
        row["loudness"] = -10.0 - i
        row[TARGET] = mood
        rows.append(row)
    return pd.DataFrame(rows, columns=[*FEATURES, TARGET])


def test_run_eda_populates_report_and_saves_plots(tmp_path) -> None:
    """run_eda returns a populated report and writes PNG plot files.

    Validates Requirements 3.1, 3.3, 3.4, 3.7.
    """
    df = _sample_dataframe()
    out_dir = tmp_path / "eda"

    report = run_eda(df, out_dir=str(out_dir))

    assert isinstance(report, EdaReport)
    # Report fields populated (3.1, 3.3, 3.4).
    assert report.row_count == len(df)
    assert report.col_count == df.shape[1]
    assert set(report.dtypes.keys()) == set(df.columns)
    assert isinstance(report.duplicate_count, int)
    assert sum(report.class_distribution.values()) == len(df)
    assert set(report.feature_summary.keys()) == set(FEATURES)

    # PNG files created in the output directory (3.7).
    for name in ("class_distribution", "feature_distributions", "correlation_matrix"):
        path = report.plot_paths[name]
        assert os.path.isfile(path)
        assert path.endswith(".png")


# ---------------------------------------------------------------------------
# Training / evaluation / model_io unit tests (Tasks 7.3, 8.2, 9.4)
# ---------------------------------------------------------------------------

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import MOOD_CLASSES
from src.evaluate import evaluate_model
from src.model_io import ModelFileNotFoundError, load_model, save_model
from src.train import PARAM_GRID, train_model


def _tiny_xy(n: int = 40):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({f: rng.uniform(0, 1, n) for f in FEATURES}, columns=FEATURES)
    moods = np.array((MOOD_CLASSES * (n // len(MOOD_CLASSES) + 1))[:n])
    encoder = LabelEncoder()
    y = encoder.fit_transform(moods)
    return X, y, encoder


def test_train_model_returns_random_forest_within_grid() -> None:
    """Trained model is a RandomForest and best_params come from the grid.

    Validates Requirements 5.1, 5.2, 5.3.
    """
    X, y, _ = _tiny_xy()
    grid = {"n_estimators": [15, 25], "max_depth": [5],
            "min_samples_split": [2], "max_features": ["sqrt"]}
    result = train_model(X, y, seed=42, param_grid=grid)

    assert isinstance(result.model, RandomForestClassifier)
    for key, value in result.best_params.items():
        assert value in grid[key]
    assert 0.0 <= result.cv_mean <= 1.0
    assert result.cv_std >= 0.0


def test_evaluate_model_metrics_and_importances() -> None:
    """Metrics are in [0,1] and importances length 8 sum to ~1.

    Validates Requirements 6.1, 6.4, 6.5.
    """
    X, y, encoder = _tiny_xy()
    model = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    report = evaluate_model(model, X, y, encoder, out_dir="outputs/evaluation")

    for metric in (report.accuracy, report.precision, report.recall, report.f1):
        assert 0.0 <= metric <= 1.0
    assert len(report.feature_importance) == 8
    assert abs(sum(report.feature_importance.values()) - 1.0) < 1e-6
    for path in report.plot_paths.values():
        assert os.path.isfile(path)


def test_load_model_missing_file_is_descriptive(tmp_path) -> None:
    """Loading a missing model file raises a descriptive error.

    Validates Requirement 7.4.
    """
    missing = str(tmp_path / "nope.pkl")
    with pytest.raises(ModelFileNotFoundError) as exc_info:
        load_model(missing)
    assert missing in str(exc_info.value)


def test_save_load_round_trip_predicts() -> None:
    """A saved-then-loaded bundle can predict a valid mood.

    Validates Requirements 7.1, 7.2, 9.3.
    """
    X, y, encoder = _tiny_xy()
    model = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    path = "models/_unit_test_model.pkl"
    save_model(model, encoder, {"note": "unit-test"}, path)
    bundle = load_model(path)
    features = {f: 0.5 for f in FEATURES}
    features["tempo"] = 120.0
    features["loudness"] = -6.0
    pred = bundle.predict(features)
    assert pred.mood in MOOD_CLASSES
    os.remove(path)

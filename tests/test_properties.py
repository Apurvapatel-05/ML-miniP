"""Property-based tests for the Music Mood Predictor.

Each test validates one of the correctness properties from the design document
using ``hypothesis`` at a minimum of 100 iterations. Generators follow the
design's Testing Strategy: DataFrames built from in-range feature rows plus
random mood labels (with injected duplicates and NaNs), and label sequences
drawn via ``st.sampled_from(MOOD_CLASSES)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.config import FEATURES, MOOD_CLASSES, TARGET
from src.data_loader import MissingColumnError, load_dataset
from src.preprocessing import encode_labels, preprocess

REQUIRED_COLUMNS = [*FEATURES, TARGET]


# ---------------------------------------------------------------------------
# Shared generators
# ---------------------------------------------------------------------------

# In-range float for each feature (loose bounds; preprocessing only requires
# numeric, finite values). We keep values finite to avoid NaN/inf artifacts.
_feature_value = st.floats(
    min_value=-100.0, max_value=300.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _feature_row(draw) -> dict:
    """Draw a single row: one float per feature plus a random mood label."""
    row = {feature: draw(_feature_value) for feature in FEATURES}
    row[TARGET] = draw(st.sampled_from(MOOD_CLASSES))
    return row


@st.composite
def _dataset(draw) -> pd.DataFrame:
    """Build a small dataset that is guaranteed to support a stratified split.

    Ensures at least two DISTINCT rows for every one of the four mood classes so
    that after duplicate removal a stratified train/test split is always
    feasible (>= 2 members per class). Optionally injects duplicate rows and NaN
    feature values to exercise the cleaning steps.
    """
    # Base rows: guarantee >= 5 distinct rows per class (20 rows total). A
    # stratified split with test_size=0.2 needs the test partition to contain at
    # least one sample per class (test count >= n_classes = 4); 20 rows yields a
    # 4-sample test set, satisfying that constraint. Each base row also gets a
    # unique 'danceability' offset so drop_duplicates cannot collapse the
    # guaranteed rows below 5-per-class even if the other features coincide.
    rows: list[dict] = []
    unique_seed = 0
    for mood in MOOD_CLASSES:
        for _ in range(5):
            row = {feature: draw(_feature_value) for feature in FEATURES}
            # Force uniqueness across all guaranteed rows.
            row["danceability"] = float(unique_seed) / 1000.0
            row[TARGET] = mood
            rows.append(row)
            unique_seed += 1

    # Extra random rows.
    extra = draw(st.lists(_feature_row(), min_size=0, max_size=12))
    rows.extend(extra)

    n_base = len(rows)  # number of guaranteed base rows (20)
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    # Inject duplicate rows (Requirement 4.1 coverage) by duplicating some base
    # rows. Duplicates are removed by preprocessing, and each class still keeps
    # its 5 distinct base rows.
    if draw(st.booleans()):
        n_dup = draw(st.integers(min_value=1, max_value=3))
        df = pd.concat([df, df.head(n_dup)], ignore_index=True)

    # Inject NaN feature values (Requirement 4.2 coverage), only into extra
    # (non-base) rows so every class retains its clean base rows for a valid
    # stratified split.
    if draw(st.booleans()) and len(df) > n_base:
        idx = draw(st.integers(min_value=n_base, max_value=len(df) - 1))
        feature = draw(st.sampled_from(FEATURES))
        df.loc[idx, feature] = np.nan

    return df


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 1: For any raw dataset containing the
# eight Audio_Features and the Target_Column (possibly with duplicate rows and
# missing values), after preprocessing completes the resulting training and test
# feature sets SHALL contain exactly the eight Audio_Features, only numeric
# values, and no missing values.
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(df=_dataset())
def test_property_1_preprocessing_output_invariants(df: pd.DataFrame) -> None:
    """Validates: Requirements 4.1, 4.2, 4.4, 4.6."""
    split = preprocess(df)

    for frame in (split.X_train, split.X_test):
        # Exactly the eight audio features, in canonical order.
        assert list(frame.columns) == FEATURES
        # Only numeric values.
        assert all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)
        # No missing values.
        assert not frame.isna().any().any()


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 2: For any sequence of Mood_Class
# labels drawn from {Happy, Sad, Calm, Energetic}, encoding the labels to
# integers and then inverse-transforming SHALL reproduce the original labels
# exactly.
@settings(max_examples=100)
@given(moods=st.lists(st.sampled_from(MOOD_CLASSES), min_size=1, max_size=50))
def test_property_2_label_encoding_round_trip(moods: list[str]) -> None:
    """Validates: Requirements 4.3."""
    series = pd.Series(moods)
    encoded, encoder = encode_labels(series)
    decoded = encoder.inverse_transform(encoded)

    assert list(decoded) == moods


# ---------------------------------------------------------------------------
# Property 3
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 3: For any dataset and a fixed random
# seed, splitting into training and test sets SHALL be reproducible (identical
# splits across runs with the same seed), and the training and test sets
# together SHALL partition all rows with no overlap.
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(df=_dataset())
def test_property_3_split_reproducibility_and_partition(df: pd.DataFrame) -> None:
    """Validates: Requirements 4.5."""
    split_a = preprocess(df, seed=42)
    split_b = preprocess(df, seed=42)

    # Reproducibility: identical splits (same rows and order) across runs.
    pd.testing.assert_frame_equal(split_a.X_train, split_b.X_train)
    pd.testing.assert_frame_equal(split_a.X_test, split_b.X_test)
    assert np.array_equal(split_a.y_train, split_b.y_train)
    assert np.array_equal(split_a.y_test, split_b.y_test)

    # Partition: train and test indices are disjoint and together cover all
    # rows that survive duplicate/missing-target cleaning.
    train_idx = set(split_a.X_train.index)
    test_idx = set(split_a.X_test.index)
    assert train_idx.isdisjoint(test_idx)

    expected_rows = len(df.drop_duplicates().dropna(subset=[TARGET]))
    assert len(train_idx) + len(test_idx) == expected_rows


# ---------------------------------------------------------------------------
# Property 4
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 4: For any dataset that is missing at
# least one required column (an Audio_Feature or the Target_Column), loading
# SHALL raise a descriptive error that names a missing column.
@settings(max_examples=100)
@given(
    # Drop between 1 and all-but-one required columns so the CSV still has at
    # least one column to parse (an entirely empty file is a distinct error
    # case, not a "missing required column" case).
    dropped=st.lists(
        st.sampled_from(REQUIRED_COLUMNS),
        min_size=1,
        max_size=len(REQUIRED_COLUMNS) - 1,
        unique=True,
    )
)
def test_property_4_missing_required_column_detection(
    tmp_path_factory, dropped: list[str]
) -> None:
    """Validates: Requirements 2.4."""
    # Build a complete dataset then drop the selected required column(s). We
    # also keep an extra non-required column so at least one column always
    # remains even if every required column were selected.
    data = {col: [0.5, 0.6] for col in FEATURES}
    data[TARGET] = ["Happy", "Sad"]
    data["_extra"] = [1, 2]
    df = pd.DataFrame(data)
    df = df.drop(columns=dropped)

    csv_path = tmp_path_factory.mktemp("data") / "missing_cols.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(MissingColumnError) as exc_info:
        load_dataset(str(csv_path))

    # The error message names at least one of the missing columns.
    message = str(exc_info.value)
    assert any(col in message for col in dropped)


# ---------------------------------------------------------------------------
# Shared helpers for model-level properties
# ---------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from src.config import FEATURE_RANGES  # noqa: E402
from src.model_io import ModelBundle, load_model, save_model  # noqa: E402
from src.validation import validate_input  # noqa: E402


def _tiny_trained_bundle() -> ModelBundle:
    """Build a small, deterministic trained bundle for model-level tests."""
    rng = np.random.RandomState(0)
    n = 40
    data = {f: rng.uniform(0, 1, n) for f in FEATURES}
    X = pd.DataFrame(data, columns=FEATURES)
    moods = np.array((MOOD_CLASSES * (n // len(MOOD_CLASSES) + 1))[:n])
    encoder = LabelEncoder()
    y = encoder.fit_transform(moods)
    model = RandomForestClassifier(n_estimators=25, random_state=0)
    model.fit(X, y)
    return ModelBundle(model=model, label_encoder=encoder,
                       feature_order=list(FEATURES), metadata={})


# In-range feature dict generator honoring FEATURE_RANGES.
_in_range_features = st.fixed_dictionaries(
    {
        f: st.floats(
            min_value=FEATURE_RANGES[f][0],
            max_value=FEATURE_RANGES[f][1],
            allow_nan=False,
            allow_infinity=False,
        )
        for f in FEATURES
    }
)


# ---------------------------------------------------------------------------
# Property 5
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 5: For any fixed training dataset and
# seed, training the Classifier twice SHALL produce models that yield identical
# predictions for identical Audio_Features input.
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(features=_in_range_features)
def test_property_5_training_reproducibility(features: dict) -> None:
    """Validates: Requirements 5.4."""
    from src.train import train_model

    rng = np.random.RandomState(1)
    n = 40
    X = pd.DataFrame({f: rng.uniform(0, 1, n) for f in FEATURES}, columns=FEATURES)
    moods = np.array((MOOD_CLASSES * (n // len(MOOD_CLASSES) + 1))[:n])
    y = LabelEncoder().fit_transform(moods)

    small_grid = {"n_estimators": [15], "max_depth": [5],
                  "min_samples_split": [2], "max_features": ["sqrt"]}
    r1 = train_model(X, y, seed=42, param_grid=small_grid)
    r2 = train_model(X, y, seed=42, param_grid=small_grid)

    row = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    assert np.array_equal(r1.model.predict(row), r2.model.predict(row))


# ---------------------------------------------------------------------------
# Property 6
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 6: For any trained Mood_Model and for
# any valid Audio_Features input, saving the model to music_mood_model.pkl and
# then loading it SHALL produce a model that yields identical predictions.
@settings(max_examples=100, deadline=None)
@given(features=_in_range_features)
def test_property_6_model_save_load_round_trip(
    features: dict, tmp_path_factory
) -> None:
    """Validates: Requirements 7.3."""
    bundle = _tiny_trained_bundle()
    path = str(tmp_path_factory.mktemp("model") / "m.pkl")
    save_model(bundle.model, bundle.label_encoder, {}, path)
    loaded = load_model(path)

    before = bundle.predict(features)
    after = loaded.predict(features)
    assert before.mood == after.mood
    assert before.confidence == after.confidence


# ---------------------------------------------------------------------------
# Property 7
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 7: For any valid in-range
# Audio_Features input, the prediction SHALL return exactly one Mood_Class from
# {Happy, Sad, Calm, Energetic} and a Confidence_Score within [0, 100] equal to
# the maximum class probability as a percentage.
@settings(max_examples=100, deadline=None)
@given(features=_in_range_features)
def test_property_7_prediction_output_contract(features: dict) -> None:
    """Validates: Requirements 1.5, 9.3, 9.4."""
    bundle = _tiny_trained_bundle()
    pred = bundle.predict(features)

    assert pred.mood in MOOD_CLASSES
    assert 0.0 <= pred.confidence <= 100.0
    assert pred.confidence == max(pred.probabilities.values())


# ---------------------------------------------------------------------------
# Property 8
# ---------------------------------------------------------------------------
# Feature: music-mood-predictor, Property 8: For any Audio_Features input in
# which at least one feature is non-numeric or out of range, validation SHALL
# return an error identifying an offending feature and withhold prediction; for
# any all-in-range numeric input, validation SHALL return no errors.
@settings(max_examples=100)
@given(
    features=_in_range_features,
    break_feature=st.sampled_from(FEATURES),
    over=st.booleans(),
)
def test_property_8_input_validation_partitions(
    features: dict, break_feature: str, over: bool
) -> None:
    """Validates: Requirements 9.7, 12.4."""
    # All in-range -> no errors.
    assert validate_input(dict(features)) == []

    # Break one feature out of range -> at least one error naming it.
    low, high = FEATURE_RANGES[break_feature]
    bad = dict(features)
    bad[break_feature] = (high + 1000.0) if over else (low - 1000.0)
    errors = validate_input(bad)
    assert errors
    assert any(e.feature == break_feature for e in errors)

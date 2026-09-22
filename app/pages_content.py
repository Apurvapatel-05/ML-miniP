"""Streamlit page rendering and theming for the Music Mood Predictor.

Implements the dark glassmorphism visual language (purple/magenta gradients,
frosted-glass cards, pill navigation, glowing accents) and the Home, Prediction,
and Analytics page content (Requirements 8-11).
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    DATA_PATH,
    FEATURE_RANGES,
    FEATURES,
    MOOD_CLASSES,
    MOOD_DESCRIPTIONS,
    TARGET,
)
from src.validation import validate_input

# Per-mood accent colors used for the result-card glow.
MOOD_COLORS: dict[str, str] = {
    "Happy": "#F4B14B",
    "Sad": "#4B79F4",
    "Calm": "#4BE0C6",
    "Energetic": "#F44B9E",
}

# Friendly help text and default values for each feature input.
FEATURE_HELP: dict[str, str] = {
    "danceability": "How suitable the track is for dancing (0-1).",
    "energy": "Perceptual intensity and activity (0-1).",
    "valence": "Musical positiveness; high = happy, low = sad (0-1).",
    "tempo": "Estimated tempo in beats per minute (BPM).",
    "loudness": "Overall loudness in decibels (dB), usually negative.",
    "acousticness": "Confidence the track is acoustic (0-1).",
    "speechiness": "Presence of spoken words (0-1).",
    "instrumentalness": "Likelihood the track has no vocals (0-1).",
}

FEATURE_DEFAULTS: dict[str, float] = {
    "danceability": 0.65,
    "energy": 0.70,
    "valence": 0.55,
    "tempo": 120.0,
    "loudness": -6.0,
    "acousticness": 0.15,
    "speechiness": 0.06,
    "instrumentalness": 0.0,
}


def inject_theme() -> None:
    """Inject the custom glassmorphism CSS used across every page."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(123,47,247,0.28), transparent 60%),
                radial-gradient(1000px 500px at 110% 10%, rgba(224,82,160,0.22), transparent 55%),
                linear-gradient(160deg, #0B0B0F 0%, #141018 100%);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 22px;
            padding: 26px 30px;
            margin-bottom: 22px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #B14BF4, #E052A0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtle { color: rgba(245,243,250,0.72); }
        .mood-pill {
            display: inline-block;
            padding: 8px 18px;
            margin: 6px 8px 6px 0;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.05);
            font-weight: 600;
        }
        .result-mood {
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0.2rem 0;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #7B2FF7, #E052A0);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.6rem 2.2rem;
            font-weight: 700;
            box-shadow: 0 4px 20px rgba(176,75,244,0.45);
            transition: transform 0.05s ease, box-shadow 0.2s ease;
        }
        div.stButton > button:hover {
            box-shadow: 0 6px 28px rgba(224,82,160,0.6);
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(html: str) -> None:
    """Render an HTML block inside a frosted glass card."""
    st.markdown(f'<div class="glass-card">{html}</div>', unsafe_allow_html=True)


def render_home() -> None:
    """Render the Home page: project intro and the four mood classes (Req 8)."""
    _card(
        '<div class="hero-title">🎵 Music Mood Predictor</div>'
        '<p class="subtle">Predict the mood of a song from its audio features '
        "using a Random Forest machine-learning model. Enter a track's "
        "characteristics and instantly see whether it feels Happy, Sad, Calm, "
        "or Energetic — along with a confidence score and insights.</p>"
    )

    pills = "".join(
        f'<span class="mood-pill" style="border-color:{MOOD_COLORS[m]};'
        f'box-shadow:0 0 14px {MOOD_COLORS[m]}55;">{m}</span>'
        for m in MOOD_CLASSES
    )
    _card(
        "<h3>Supported moods</h3>"
        f"<div>{pills}</div>"
        '<p class="subtle" style="margin-top:14px;">Head to the '
        "<b>Prediction</b> page to classify a song, or explore the "
        "<b>Analytics</b> page for model and dataset insights.</p>"
    )

    _card(
        "<h3>How it works</h3>"
        '<ol class="subtle">'
        "<li>Enter 8 audio features (danceability, energy, valence, tempo, "
        "loudness, acousticness, speechiness, instrumentalness).</li>"
        "<li>The trained Random Forest model predicts the most likely mood.</li>"
        "<li>You get the predicted mood, a confidence score, a description, "
        "and a summary of your inputs.</li>"
        "</ol>"
    )


def render_prediction(bundle: Any) -> None:
    """Render the Prediction page (Requirements 9.1-9.7, 12.2, 12.4)."""
    _card(
        '<div class="hero-title" style="font-size:2rem;">Mood Prediction</div>'
        '<p class="subtle">Enter the song\'s audio features and press Predict.</p>'
    )

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        cols = st.columns(2)
        values: dict[str, float] = {}
        for i, feature in enumerate(FEATURES):
            low, high = FEATURE_RANGES[feature]
            with cols[i % 2]:
                values[feature] = st.number_input(
                    label=f"{feature.capitalize()}  ({low} to {high})",
                    min_value=float(low),
                    max_value=float(high),
                    value=float(FEATURE_DEFAULTS[feature]),
                    step=0.01 if high <= 1.0 else 1.0,
                    help=FEATURE_HELP[feature],
                    key=f"in_{feature}",
                )
        predict_clicked = st.button("Predict Mood", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not predict_clicked:
        return

    # Validate before predicting (Req 9.7, 12.4).
    errors = validate_input(values)
    if errors:
        for err in errors:
            st.error(f"{err.feature}: {err.message}")
        return

    try:
        prediction = bundle.predict(values)
    except Exception as exc:  # log + surface (Req 12.2)
        st.error(f"Prediction failed: {exc}")
        return

    color = MOOD_COLORS.get(prediction.mood, "#B14BF4")
    _card(
        f'<p class="subtle">Predicted mood</p>'
        f'<div class="result-mood" style="color:{color};'
        f'text-shadow:0 0 22px {color}88;">{prediction.mood}</div>'
        f'<p class="subtle">Confidence: <b>{prediction.confidence:.2f}%</b></p>'
        f'<p style="margin-top:10px;">{MOOD_DESCRIPTIONS.get(prediction.mood, "")}</p>'
    )

    # Confidence across classes (Req 10-style insight on the prediction).
    proba_df = pd.DataFrame(
        {"Mood": list(prediction.probabilities.keys()),
         "Confidence (%)": list(prediction.probabilities.values())}
    ).set_index("Mood")
    st.bar_chart(proba_df)

    # Audio feature summary of the submitted values (Req 9.6).
    summary = pd.DataFrame(
        {"Feature": FEATURES, "Value": [values[f] for f in FEATURES]}
    ).set_index("Feature")
    _card("<h4>Your audio feature summary</h4>")
    st.table(summary)


def render_analytics(bundle: Any) -> None:
    """Render the Analytics page (Requirements 10.1-10.3, 11.2)."""
    _card(
        '<div class="hero-title" style="font-size:2rem;">Analytics</div>'
        '<p class="subtle">Model insights and dataset characteristics.</p>'
    )

    # Feature importance from model metadata or the model itself (Req 10.1).
    importance = bundle.metadata.get("feature_importance") if bundle.metadata else None
    if not importance and hasattr(bundle.model, "feature_importances_"):
        importance = {
            f: float(v) for f, v in zip(FEATURES, bundle.model.feature_importances_)
        }
    if importance:
        _card("<h4>Feature importance</h4>")
        imp_df = pd.DataFrame(
            {"Feature": list(importance.keys()),
             "Importance": list(importance.values())}
        ).set_index("Feature").sort_values("Importance", ascending=False)
        st.bar_chart(imp_df)

    # Saved evaluation plot (with caption/alt text for accessibility, Req 11.2).
    fi_plot = "outputs/evaluation/feature_importance.png"
    if os.path.isfile(fi_plot):
        st.image(fi_plot, caption="Feature importance (Random Forest)",
                 use_container_width=True)

    # Model performance metrics from metadata (Req 10.3).
    md = bundle.metadata or {}
    if md:
        _card("<h4>Model performance</h4>")
        metrics = {
            "Test accuracy": md.get("test_accuracy"),
            "Macro F1": md.get("test_f1"),
            "CV accuracy (mean)": md.get("cv_mean"),
            "CV accuracy (std)": md.get("cv_std"),
        }
        mcols = st.columns(len(metrics))
        for col, (name, val) in zip(mcols, metrics.items()):
            col.metric(name, f"{val:.3f}" if isinstance(val, (int, float)) else "n/a")

    # Dataset statistics incl. class distribution (Req 10.2).
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        _card("<h4>Dataset statistics</h4>")
        c1, c2 = st.columns(2)
        c1.metric("Total songs", f"{len(df):,}")
        c2.metric("Audio features", str(len(FEATURES)))

        dist = df[TARGET].value_counts()
        dist_df = pd.DataFrame({"Mood": dist.index, "Count": dist.values}).set_index("Mood")
        st.markdown("**Mood class distribution**")
        st.bar_chart(dist_df)

        # Saved EDA plots for richer visual insight (Req 10.3, 11.2).
        for name, cap in [
            ("outputs/eda/class_distribution.png", "Mood class distribution"),
            ("outputs/eda/correlation_matrix.png", "Audio feature correlation matrix"),
            ("outputs/eda/feature_distributions.png", "Audio feature distributions"),
        ]:
            if os.path.isfile(name):
                st.image(name, caption=cap, use_container_width=True)

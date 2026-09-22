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

# Per-mood emoji used in the result reveal for a friendlier, more vivid UI.
MOOD_EMOJI: dict[str, str] = {
    "Happy": "😄",
    "Sad": "😢",
    "Calm": "😌",
    "Energetic": "⚡",
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
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

        /* Animated aurora background */
        .stApp {
            background:
                radial-gradient(1100px 600px at 12% -12%, rgba(123,47,247,0.32), transparent 60%),
                radial-gradient(1000px 520px at 112% 8%, rgba(224,82,160,0.26), transparent 55%),
                radial-gradient(900px 500px at 50% 120%, rgba(75,224,198,0.14), transparent 55%),
                linear-gradient(160deg, #08070C 0%, #120E1A 55%, #0B0910 100%);
            background-attachment: fixed;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: -20%;
            z-index: -1;
            background:
                radial-gradient(600px 600px at 20% 30%, rgba(123,47,247,0.20), transparent 60%),
                radial-gradient(600px 600px at 80% 70%, rgba(224,82,160,0.18), transparent 60%);
            filter: blur(40px);
            animation: drift 18s ease-in-out infinite alternate;
        }
        @keyframes drift {
            0%   { transform: translate(0, 0) scale(1); }
            100% { transform: translate(-4%, 4%) scale(1.12); }
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .glass-card {
            background: rgba(255, 255, 255, 0.055);
            backdrop-filter: blur(20px) saturate(140%);
            -webkit-backdrop-filter: blur(20px) saturate(140%);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 24px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45),
                        inset 0 1px 0 rgba(255,255,255,0.08);
            position: relative;
            overflow: hidden;
        }
        /* Subtle top sheen on cards */
        .glass-card::after {
            content: "";
            position: absolute;
            top: 0; left: -30%;
            width: 60%; height: 100%;
            background: linear-gradient(120deg, transparent, rgba(255,255,255,0.06), transparent);
            transform: skewX(-20deg);
        }

        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.9rem;
            font-weight: 700;
            line-height: 1.1;
            background: linear-gradient(90deg, #9B6BFF 0%, #E052A0 50%, #4BE0C6 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 6s linear infinite;
            margin-bottom: 0.35rem;
        }
        @keyframes shimmer {
            to { background-position: 200% center; }
        }
        .subtle { color: rgba(245,243,250,0.74); line-height: 1.6; }

        h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.2px; }

        /* Glowing hero orb (spinning conic gradient with a glass core) */
        .orb-wrap { display: flex; justify-content: center; margin: 8px 0 4px; }
        .orb {
            width: 150px; height: 150px; border-radius: 50%;
            background: conic-gradient(from 0deg, #7B2FF7, #E052A0, #4BE0C6, #7B2FF7);
            animation: spin 7s linear infinite;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 50px rgba(176,75,244,0.55), 0 0 90px rgba(224,82,160,0.35);
        }
        .orb-core {
            width: 116px; height: 116px; border-radius: 50%;
            background: rgba(10,8,16,0.82);
            backdrop-filter: blur(8px);
            display: flex; align-items: center; justify-content: center;
            font-size: 3rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .mood-pill {
            display: inline-block;
            padding: 9px 20px;
            margin: 6px 8px 6px 0;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.05);
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.2s ease;
        }
        .mood-pill:hover { transform: translateY(-2px); }

        .result-mood {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            margin: 0.15rem 0;
            letter-spacing: 0.5px;
        }
        .result-card { animation: rise 0.5s cubic-bezier(0.2, 0.8, 0.2, 1); }
        @keyframes rise {
            from { opacity: 0; transform: translateY(14px) scale(0.98); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        .conf-track {
            width: 100%; height: 12px; border-radius: 999px;
            background: rgba(255,255,255,0.10); overflow: hidden; margin: 10px 0 4px;
        }
        .conf-fill {
            height: 100%; border-radius: 999px;
            animation: grow 0.9s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        @keyframes grow { from { width: 0; } }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #7B2FF7, #E052A0);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 2.4rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.4px;
            box-shadow: 0 6px 24px rgba(176,75,244,0.5);
            transition: transform 0.06s ease, box-shadow 0.25s ease, filter 0.2s ease;
        }
        div.stButton > button:hover {
            box-shadow: 0 10px 34px rgba(224,82,160,0.7);
            transform: translateY(-2px);
            filter: brightness(1.08);
        }
        div.stButton > button:active { transform: translateY(0); }

        /* Inputs */
        div[data-testid="stNumberInput"] input {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.12);
            color: #F5F3FA;
        }
        div[data-testid="stNumberInput"] input:focus {
            border-color: #B14BF4;
            box-shadow: 0 0 0 2px rgba(176,75,244,0.35);
        }

        /* Metric tiles */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 6px 22px rgba(0,0,0,0.30);
        }
        div[data-testid="stMetricValue"] {
            background: linear-gradient(90deg, #B14BF4, #E052A0);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Space Grotesk', sans-serif;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(20,16,26,0.9), rgba(11,9,16,0.9));
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* Floating music notes */
        .note {
            position: fixed; bottom: -40px; color: rgba(255,255,255,0.12);
            font-size: 1.5rem; z-index: -1; animation: float 14s linear infinite;
        }
        .note.n1 { left: 8%;  animation-delay: 0s;  }
        .note.n2 { left: 30%; animation-delay: 4s;  font-size: 1.1rem; }
        .note.n3 { left: 62%; animation-delay: 2s;  font-size: 2rem; }
        .note.n4 { left: 85%; animation-delay: 7s;  }
        @keyframes float {
            0%   { transform: translateY(0) rotate(0);   opacity: 0; }
            10%  { opacity: 1; }
            90%  { opacity: 1; }
            100% { transform: translateY(-110vh) rotate(40deg); opacity: 0; }
        }
        </style>
        <div class="note n1">♪</div>
        <div class="note n2">♫</div>
        <div class="note n3">♩</div>
        <div class="note n4">♬</div>
        """,
        unsafe_allow_html=True,
    )


def _card(html: str) -> None:
    """Render an HTML block inside a frosted glass card."""
    st.markdown(f'<div class="glass-card">{html}</div>', unsafe_allow_html=True)


def render_home() -> None:
    """Render the Home page: project intro and the four mood classes (Req 8)."""
    _card(
        '<div class="orb-wrap"><div class="orb"><div class="orb-core">🎧</div></div></div>'
        '<div class="hero-title" style="text-align:center;">Music Mood Predictor</div>'
        '<p class="subtle" style="text-align:center;max-width:640px;margin:0 auto;">'
        "Predict the mood of a song from its audio features using a Random Forest "
        "machine-learning model. Enter a track's characteristics and instantly see "
        "whether it feels Happy, Sad, Calm, or Energetic — along with a confidence "
        "score and rich insights.</p>"
    )

    pills = "".join(
        f'<span class="mood-pill" style="border-color:{MOOD_COLORS[m]};'
        f'box-shadow:0 0 16px {MOOD_COLORS[m]}55;color:{MOOD_COLORS[m]};">'
        f'{MOOD_EMOJI[m]} {m}</span>'
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
    emoji = MOOD_EMOJI.get(prediction.mood, "🎵")
    st.markdown(
        f'<div class="glass-card result-card" '
        f'style="border-color:{color}66;box-shadow:0 10px 40px {color}33,'
        f'inset 0 1px 0 rgba(255,255,255,0.08);">'
        f'<p class="subtle" style="margin-bottom:2px;">Predicted mood</p>'
        f'<div class="result-mood" style="color:{color};'
        f'text-shadow:0 0 26px {color}99;">{emoji}&nbsp;{prediction.mood}</div>'
        f'<div class="conf-track"><div class="conf-fill" '
        f'style="width:{prediction.confidence:.1f}%;'
        f'background:linear-gradient(90deg,{color},#ffffff88);"></div></div>'
        f'<p class="subtle">Confidence: <b style="color:{color};">'
        f'{prediction.confidence:.2f}%</b></p>'
        f'<p style="margin-top:10px;">{MOOD_DESCRIPTIONS.get(prediction.mood, "")}</p>'
        f"</div>",
        unsafe_allow_html=True,
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

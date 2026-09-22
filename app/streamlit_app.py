"""Streamlit entry point for the Music Mood Predictor.

Provides navigation across Home, Prediction, and Analytics pages, loads the
trained model bundle once (cached), and applies the dark glassmorphism theme.
Run with:  streamlit run app/streamlit_app.py

Requirements: 7.2, 7.4, 8.3, 11.1.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure the project root is importable when Streamlit runs this file directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.pages_content import (  # noqa: E402
    inject_theme,
    render_analytics,
    render_home,
    render_prediction,
)
from src.config import MODEL_PATH  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.model_io import ModelFileNotFoundError, load_model  # noqa: E402

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def _get_bundle():
    """Load the model bundle once and cache it for the session."""
    return load_model(MODEL_PATH)


def main() -> None:
    """Configure the page, apply the theme, and route to the selected page."""
    st.set_page_config(
        page_title="Music Mood Predictor",
        page_icon="🎵",
        layout="wide",
    )
    inject_theme()

    # Pill-style navigation (Req 8.3).
    with st.sidebar:
        st.markdown("### 🎵 Navigation")
        page = st.radio(
            "Go to",
            ["Home", "Prediction", "Analytics"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Random Forest · Streamlit · scikit-learn")

    # Home needs no model; the other pages do.
    if page == "Home":
        render_home()
        return

    try:
        bundle = _get_bundle()
    except ModelFileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "The model file is missing. Run `python run_pipeline.py` to train "
            "and generate `music_mood_model.pkl`, then reload this page."
        )
        logger.error("Model file missing at %s", MODEL_PATH)
        return
    except Exception as exc:  # defensive: surface any load failure (Req 12.2)
        st.error(f"Failed to load the model: {exc}")
        logger.exception("Unexpected error loading model")
        return

    if page == "Prediction":
        render_prediction(bundle)
    elif page == "Analytics":
        render_analytics(bundle)


if __name__ == "__main__":
    main()

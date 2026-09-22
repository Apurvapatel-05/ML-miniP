# Project Plan — Music Mood Predictor

## Problem Statement

Listeners and music platforms often want to organize or recommend songs by
*mood*, but mood is subjective and not directly available as metadata. This
project builds a machine-learning system that predicts a song's mood from
objective audio features, turning eight numeric characteristics into one of
four interpretable mood classes.

## Objectives

1. Train a reliable **Random Forest Classifier** to predict song mood.
2. Classify each song as **Happy, Sad, Calm, or Energetic**.
3. Deliver an interactive, professional **Streamlit** web application.
4. Provide analytics and model transparency (feature importance, metrics).
5. Ship a clean, documented, deployable, portfolio-quality project.

## Scope

**In scope:** data analysis, preprocessing, model training/tuning/evaluation,
model persistence, a 3-page web app (Home, Prediction, Analytics), logging,
input validation, tests, documentation, and public deployment.

**Out of scope:** deep-learning models, multiple competing algorithms, audio
file ingestion / feature extraction from raw audio, and user accounts.

## Technology Stack

- **Language:** Python 3.11
- **Data:** pandas, numpy
- **ML:** scikit-learn (RandomForestClassifier, GridSearchCV, StratifiedKFold)
- **Plots:** matplotlib, seaborn
- **Persistence:** joblib
- **Web app:** Streamlit
- **Testing:** pytest, hypothesis (property-based)

## Development Roadmap

| Phase | Deliverable | Status |
|------|-------------|--------|
| 1 | Project planning & structure | ✅ Done |
| 2 | Data analysis & EDA | ✅ Done |
| 3 | Data preprocessing | ✅ Done |
| 4 | Model training, tuning & evaluation | ✅ Done |
| 5 | Streamlit web application | ✅ Done |
| 6 | Production readiness (logging, validation, deps) | ✅ Done |
| 7 | GitHub repository setup | ✅ Done |
| 8 | Professional README | ✅ Done |
| 9 | Public deployment (live URL) | ✅ Done |

**Live app:** https://ml-minip-ez2gz7r4jtn6wsglxehbec.streamlit.app/

## Environment Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) retrain the model — regenerates music_mood_model.pkl + plots
python run_pipeline.py

# 4. Launch the web app
streamlit run app/streamlit_app.py

# 5. Run tests
python -m pytest tests/ -q
```

# Implementation Plan: Music Mood Predictor

## Overview

This plan converts the approved design into an incremental, test-driven build in **Python**. Work proceeds bottom-up: dataset ingestion → shared config/logger → data loading → EDA → preprocessing → training → evaluation → persistence → validation → pipeline orchestration → Streamlit Web_App (dark glassmorphism theme) → production hardening → tests → repo setup/docs → deployment. Each task builds on the previous ones and wires into the growing system so no code is left orphaned.

Property-based tests (Properties 1–8 from the design) use `hypothesis` at a minimum of 100 iterations and are placed close to the code they validate. Test-related sub-tasks are marked optional with `*`.

## Tasks

- [ ] 1. Ingest dataset and scaffold project structure
  - Copy the user's dataset from `C:/Users/Soham/Downloads/spotify_labeled (1).csv` into the workspace as `data/spotify_labeled.csv` (the agent cannot read outside the workspace, so perform the copy via a shell command)
  - Verify the copied file's schema: confirm columns `danceability, energy, valence, tempo, loudness, acousticness, speechiness, instrumentalness, mood` are present, and that `mood` contains only the 4 classes {Happy, Sad, Calm, Energetic}
  - Create the modular folder structure: `src/`, `app/`, `models/`, `outputs/eda/`, `outputs/evaluation/`, `docs/`, `tests/`, `logs/` with `src/__init__.py`
  - _Requirements: 1.4, 2.1, 12.1, 13.3_

- [ ] 2. Implement shared configuration and logging foundation
  - [ ] 2.1 Implement `src/config.py`
    - Define `FEATURES` (8 audio features), `TARGET`, `MOOD_CLASSES`, `FEATURE_RANGES`, `RANDOM_SEED=42`, `TEST_SIZE=0.2`, `DATA_PATH`, `MODEL_PATH`, and `MOOD_DESCRIPTIONS` (per-class text)
    - _Requirements: 1.5, 4.4, 4.5, 5.4, 9.5, 12.4_
  - [ ] 2.2 Implement `src/logger.py`
    - Provide `get_logger(name)` with format `%(asctime)s | %(levelname)s | %(name)s | %(message)s` (timestamp + severity), console + `logs/app.log` handlers
    - _Requirements: 12.2, 12.3_
  - [ ]* 2.3 Write unit test for logger record format
    - Assert emitted records carry a timestamp and severity level
    - _Requirements: 12.3_

- [ ] 3. Implement data loading with column validation
  - [ ] 3.1 Implement `src/data_loader.py`
    - `load_dataset(path=DATA_PATH) -> pd.DataFrame`; raise `FileNotFoundError` naming the path when absent
    - Define `MissingColumnError`; after read, validate all 8 features + target present and raise naming the missing column(s)
    - _Requirements: 2.2, 2.3, 2.4, 12.2_
  - [ ]* 3.2 Write property test for missing required column detection
    - **Property 4: Missing required column detection**
    - **Validates: Requirements 2.4**
    - `# Feature: music-mood-predictor, Property 4`, `@settings(max_examples=100)`
  - [ ]* 3.3 Write unit tests for data loader examples/edge cases
    - Loads expected columns (2.2); missing-file error names the path (2.3)
    - _Requirements: 2.2, 2.3_

- [ ] 4. Implement EDA module with saved plots
  - [ ] 4.1 Implement `src/eda.py`
    - `run_eda(df, out_dir="outputs/eda") -> EdaReport` computing row/col counts, dtypes, missing counts, duplicate count, class distribution, per-feature summary stats
    - Generate and save PNG plots: mood class distribution, audio feature distributions, feature correlation matrix; note class imbalance in report
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_
  - [ ]* 4.2 Write unit test for EDA report + plot outputs
    - Report fields populated; PNG files created in `outputs/eda/`
    - _Requirements: 3.1, 3.3, 3.4, 3.7_

- [ ] 5. Implement preprocessing module
  - [ ] 5.1 Implement `src/preprocessing.py`
    - `encode_labels`, `validate_features`, and `preprocess(df, test_size, seed) -> SplitData`
    - Remove duplicates; drop rows missing target; median-impute missing features; label-encode `mood`; select 8 features; stratified split with fixed seed; post-split validate (exactly 8 numeric features, no missing values)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 12.2_
  - [ ]* 5.2 Write property test for preprocessing output invariants
    - **Property 1: Preprocessing output invariants**
    - **Validates: Requirements 4.1, 4.2, 4.4, 4.6**
    - `# Feature: music-mood-predictor, Property 1`, `@settings(max_examples=100)`
  - [ ]* 5.3 Write property test for label encoding round-trip
    - **Property 2: Label encoding round-trip**
    - **Validates: Requirements 4.3**
    - `# Feature: music-mood-predictor, Property 2`, `@settings(max_examples=100)`
  - [ ]* 5.4 Write property test for split reproducibility and partition
    - **Property 3: Split reproducibility and partition**
    - **Validates: Requirements 4.5**
    - `# Feature: music-mood-predictor, Property 3`, `@settings(max_examples=100)`

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement model training with tuning and cross-validation
  - [ ] 7.1 Implement `src/train.py`
    - `train_model(X_train, y_train, seed) -> TrainResult` using `RandomForestClassifier(class_weight="balanced", random_state=seed)` only
    - `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `max_features`; `StratifiedKFold` CV reporting mean/std accuracy; return fitted model, best_params, cv_mean, cv_std
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [ ]* 7.2 Write property test for training reproducibility with fixed seed
    - **Property 5: Training reproducibility with fixed seed**
    - **Validates: Requirements 5.4**
    - Use a small grid / reduced estimators for speed; `# Feature: music-mood-predictor, Property 5`, `@settings(max_examples=100)`
  - [ ]* 7.3 Write unit tests for training examples
    - Model is a `RandomForestClassifier` (5.1); `best_params` within grid (5.2); `cv_mean ∈ [0,1]`, `cv_std ≥ 0` (5.3)
    - _Requirements: 5.1, 5.2, 5.3_

- [ ] 8. Implement model evaluation with saved plots
  - [ ] 8.1 Implement `src/evaluate.py`
    - `evaluate_model(model, X_test, y_test, label_encoder, out_dir="outputs/evaluation") -> EvalReport`
    - Compute accuracy, precision, recall, F1 (macro + per class); confusion matrix over 4 classes; per-class classification report; feature importances for 8 features; save confusion matrix and feature importance PNGs
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ]* 8.2 Write unit tests for evaluation outputs
    - Metrics in `[0,1]`; feature importances length 8 summing to ~1; PNG files created
    - _Requirements: 6.1, 6.4, 6.5_

- [ ] 9. Implement model persistence (save/load bundle)
  - [ ] 9.1 Implement `src/model_io.py`
    - `save_model(model, label_encoder, metadata, path=MODEL_PATH)` persisting bundle `{model, label_encoder, feature_order, metadata}` via joblib to `music_mood_model.pkl`
    - `load_model(path)` returning `ModelBundle` with descriptive error if file missing (`ModelFileNotFoundError`); `ModelBundle.predict(features: dict) -> Prediction` returning mood, confidence (0–100 = max class prob), full probability map
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 1.5, 9.3, 9.4_
  - [ ]* 9.2 Write property test for model save/load round-trip
    - **Property 6: Model save/load round-trip**
    - **Validates: Requirements 7.3**
    - `# Feature: music-mood-predictor, Property 6`, `@settings(max_examples=100)`
  - [ ]* 9.3 Write property test for prediction output contract
    - **Property 7: Prediction output contract**
    - **Validates: Requirements 1.5, 9.3, 9.4**
    - `# Feature: music-mood-predictor, Property 7`, `@settings(max_examples=100)`
  - [ ]* 9.4 Write unit test for missing model file error
    - Missing model file yields a descriptive error
    - _Requirements: 7.4_

- [ ] 10. Implement shared input validation
  - [ ] 10.1 Implement `src/validation.py`
    - `validate_input(values: dict[str, float]) -> list[ValidationError]` checking each of the 8 features is numeric and within `FEATURE_RANGES`; empty list = valid
    - _Requirements: 9.7, 12.4_
  - [ ]* 10.2 Write property test for input validation partitioning
    - **Property 8: Input validation partitions valid and invalid inputs**
    - **Validates: Requirements 9.7, 12.4**
    - `# Feature: music-mood-predictor, Property 8`, `@settings(max_examples=100)`

- [ ] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement pipeline orchestration and generate the model artifact
  - [ ] 12.1 Implement `run_pipeline.py`
    - Orchestrate load → EDA → preprocess → train → evaluate → save; log each stage; wrap operations in try/except that logs and surfaces descriptive errors
    - Run it to produce `outputs/eda/*`, `outputs/evaluation/*`, and `models/music_mood_model.pkl` (with metadata: cv_mean/std, best_params, test metrics, trained_at, sklearn_version)
    - _Requirements: 3.7, 6.5, 7.1, 12.1, 12.2_

- [ ] 13. Build Streamlit theme (dark glassmorphism)
  - [ ] 13.1 Create `.streamlit/config.toml`
    - Dark base theme, purple/magenta primary color (`#7B2FF7`→`#E052A0`), near-black backgrounds (`#0B0B0F`–`#141018`), clean sans-serif font
    - _Requirements: 11.1_
  - [ ] 13.2 Implement injected custom CSS in `app/pages_content.py`
    - Frosted-glass cards (`backdrop-filter: blur`, 1px light borders, 16–24px radii), gradient buttons with glowing focus rings, rounded pill navigation, mood-colored result-card glow; keep contrast accessibility compliant
    - _Requirements: 11.1, 11.2_

- [ ] 14. Build Streamlit Web_App pages
  - [ ] 14.1 Implement `app/streamlit_app.py` entry point + navigation
    - Pill-style navigation for Home / Prediction / Analytics; load `ModelBundle` once via `@st.cache_resource`; show descriptive error banner if model file missing; apply theme/CSS across all pages
    - _Requirements: 7.2, 7.4, 8.3, 11.1_
  - [ ] 14.2 Implement Home page in `app/pages_content.py`
    - Project title + mood prediction description; display the 4 Mood_Class values
    - _Requirements: 8.1, 8.2, 11.1_
  - [ ] 14.3 Implement Prediction page in `app/pages_content.py`
    - One input widget per feature (8 inputs) with range hints; predict button runs `validate_input`; on error show field-specific messages and withhold prediction; on success show predicted Mood_Class, Confidence_Score (%), mood description, and submitted-values summary; wrap in try/except that logs and renders `st.error`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 11.1, 12.2, 12.4_
  - [ ] 14.4 Implement Analytics page in `app/pages_content.py`
    - Display feature importance of 8 features; dataset statistics including Mood_Class distribution; charts with captions/alt text
    - _Requirements: 10.1, 10.2, 10.3, 11.1, 11.2_
  - [ ]* 14.5 Write unit tests for page render helpers and navigation
    - Home/Prediction/Analytics helpers return expected content; navigation reaches all three pages
    - _Requirements: 8.1, 8.2, 8.3, 10.1, 10.2_

- [ ] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Production readiness and dependencies
  - [ ] 16.1 Ensure logging + error handling wired across all modules
    - Verify data/model/prediction operations log via Logger and surface descriptive messages at operation boundaries
    - _Requirements: 12.1, 12.2, 12.3_
  - [ ] 16.2 Create `requirements.txt` with pinned versions
    - Pin pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit, pytest, hypothesis to exact versions per the design's technology stack
    - _Requirements: 12.5_
  - [ ] 16.3 Write `docs/project_plan.md` and setup instructions
    - Problem statement, objectives, scope, development roadmap; environment setup/run instructions
    - _Requirements: 1.1, 1.2, 1.3, 12.6_

- [ ] 17. Configure Git repository and documentation
  - [ ] 17.1 Create `.gitignore` and `LICENSE`
    - `.gitignore` excludes venvs, caches, logs, and generated artifacts; add a `LICENSE` file
    - _Requirements: 13.1, 13.2_
  - [ ] 17.2 Write `README.md` (all required sections)
    - Project overview, feature list, technology stack, dataset info; **system architecture + workflow/architecture diagram**; installation/local setup/usage; model info + recorded evaluation metrics; screenshot placeholders (Home/Prediction/Analytics); future enhancements + author sections
    - **Workflow structure + development roadmap** section (workflow/architecture diagram and roadmap)
    - **Live demo section with the deployed live web app URL(s)** (filled in after task 18)
    - _Requirements: 1.1, 1.2, 13.2, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_
  - [ ] 17.3 Initialize repository with a meaningful commit
    - Ensure modular folder structure committed with a descriptive commit message
    - _Requirements: 13.3, 13.4_

- [ ] 18. Deploy publicly and record the live URL (requires user's GitHub/Streamlit accounts — user action needed)
  - Write deployment documentation for Streamlit Community Cloud (preferred) with Render fallback steps
  - With the user, deploy to Streamlit Community Cloud (or Render if unavailable), obtain the public live URL, and record it in the README live demo section
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 14.6_

- [ ] 19. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Sub-tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each correctness property (1–8) is implemented by a single `hypothesis` property-based test at `@settings(max_examples=100)`, tagged `# Feature: music-mood-predictor, Property N: ...`, placed next to the code it validates.
- Task 1 brings the external dataset into the workspace since files outside the workspace cannot be read directly.
- Task 18 requires the user's GitHub/Streamlit (or Render) accounts and manual deployment; the resulting live URL is recorded in the README.
- Every task references the specific requirements it implements for full traceability across all 15 requirements.

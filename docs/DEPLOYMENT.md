# Deployment Guide — Music Mood Predictor

The app is a standard Streamlit application and can be deployed publicly for
free. **Streamlit Community Cloud** is preferred; **Render** is the alternative.

---

## Option A — Streamlit Community Cloud (preferred)

**Prerequisites:** a GitHub account and a free Streamlit Community Cloud account
(https://streamlit.io/cloud), signed in with GitHub.

### Steps

1. **Push the project to a public GitHub repository.**
   ```bash
   git init
   git add .
   git commit -m "Music Mood Predictor: end-to-end ML app"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/music-mood-predictor.git
   git push -u origin main
   ```

2. **Ensure these are committed** (they are in this repo):
   - `app/streamlit_app.py` (the entry point)
   - `requirements.txt` (pinned dependencies)
   - `models/music_mood_model.pkl` (the trained model)
   - `data/spotify_labeled.csv`, `src/`, `.streamlit/config.toml`

3. **Create the app on Streamlit Cloud:**
   - Go to https://share.streamlit.io → **New app**.
   - Select your repository, branch `main`, and set
     **Main file path** to `app/streamlit_app.py`.
   - Click **Deploy**.

4. **Get the live URL.** Streamlit builds the environment from
   `requirements.txt` and serves the app at a URL like
   `https://your-app-name.streamlit.app`.

5. **Record the URL** in `README.md` (Live Demo section and the badge at the top).

> Tip: if the build fails, check the logs in the Streamlit Cloud dashboard —
> almost always a dependency version. The pinned `requirements.txt` here is
> known-good.

---

## Option B — Render (alternative)

**Prerequisites:** a GitHub account and a free Render account
(https://render.com).

### Steps

1. Push the repo to GitHub (same as Option A, step 1).

2. On Render: **New → Web Service** → connect your GitHub repo.

3. Configure the service:
   - **Environment:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```
     streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
     ```

4. Click **Create Web Service**. Render builds and deploys, then gives a public
   URL like `https://music-mood-predictor.onrender.com`.

5. Record the URL in `README.md`.

---

## Post-deployment checklist

- [ ] App loads without errors at the public URL.
- [ ] Home, Prediction, and Analytics pages all render.
- [ ] A sample prediction returns a mood + confidence.
- [ ] Live URL added to `README.md` (Live Demo + badge).
- [ ] (Optional) Screenshots added to `docs/screenshots/`.

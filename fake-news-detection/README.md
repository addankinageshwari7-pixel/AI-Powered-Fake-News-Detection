# AI-Powered Fake News Detection System

Production-ready Flask + Scikit-learn web app that classifies news as **Real** or **Fake** using TF-IDF + Logistic Regression, with confidence, risk level, sentiment, keyword extraction, and an analytics dashboard.

## Features
- Real vs Fake classification with confidence score
- Risk level (Low / Medium / High) and Authenticity score
- Explainable AI reasoning (top contributing keywords)
- URL article extraction & analysis
- JSON API endpoint (`/api/predict`)
- Analytics dashboard with Chart.js
- Dark, professional, glassmorphism UI (Bootstrap 5)
- Zero animations, fast loading

## Quick Start
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

The model auto-trains from `dataset/news.csv` on first launch and is saved to `model/`.

## Deploy (Render / Railway)
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`
- Procfile included.

## Tech
Python · Flask · Scikit-learn · TF-IDF · Logistic Regression · Passive Aggressive Classifier · Bootstrap 5 · Chart.js

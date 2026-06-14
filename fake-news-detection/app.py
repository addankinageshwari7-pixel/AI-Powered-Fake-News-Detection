"""
AI-Powered Fake News Detection System
Flask + Scikit-learn (TF-IDF + Logistic Regression)
"""
import os
import re
import json
from datetime import datetime
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "news.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
PA_MODEL_PATH = os.path.join(MODEL_DIR, "pa_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

STOPWORDS = set("""a an the and or but if while of in on at to for with by from is are was were be been being
this that these those it its as into about over under again further then once here there when where why how
all any both each few more most other some such no nor not only own same so than too very can will just
don should now i me my we our you your he him his she her they them their what which who whom""".split())

POSITIVE_WORDS = {"good", "great", "success", "improve", "growth", "win", "agree", "support", "benefit",
                  "achievement", "progress", "approved", "confirmed", "launched", "official"}
NEGATIVE_WORDS = {"shocking", "breaking", "exposed", "secret", "hidden", "conspiracy", "leaked", "miracle",
                  "cure", "stunned", "banned", "hate", "furious", "scandal", "die", "crisis", "attack"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

# Lightweight in-memory analytics (resets on restart)
ANALYTICS = {
    "total": 0,
    "real": 0,
    "fake": 0,
    "confidence_buckets": {"50-60": 0, "60-70": 0, "70-80": 0, "80-90": 0, "90-100": 0},
    "recent": [],  # last 10 predictions
}

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def top_keywords(text: str, n: int = 8):
    tokens = clean_text(text).split()
    common = [w for w, _ in Counter(tokens).most_common(n)]
    return common


def sentiment_score(text: str):
    tokens = set(clean_text(text).split())
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    if pos == neg == 0:
        return "Neutral", 50
    score = int(round((pos / max(pos + neg, 1)) * 100))
    label = "Positive" if score >= 60 else "Negative" if score <= 40 else "Neutral"
    return label, score


def extract_article_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (FakeNewsDetector/1.0)"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if len(p) > 40)
    if not text:
        text = soup.get_text(" ", strip=True)
    return text[:8000]


# ---------------------------------------------------------------------------
# Model training / loading
# ---------------------------------------------------------------------------
def train_and_save():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["clean"] = df["text"].astype(str).apply(clean_text)
    df["target"] = df["label"].str.upper().map({"REAL": 1, "FAKE": 0})
    df = df.dropna(subset=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, max_features=5000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=4.0)
    lr.fit(Xtr, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(Xte))

    pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    pa.fit(Xtr, y_train)
    pa_acc = accuracy_score(y_test, pa.predict(Xte))

    joblib.dump(lr, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(pa, PA_MODEL_PATH)

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "samples": int(len(df)),
        "lr_accuracy": round(float(lr_acc), 4),
        "pa_accuracy": round(float(pa_acc), 4),
        "features": int(len(vectorizer.get_feature_names_out())),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return lr, vectorizer, pa, meta


def load_models():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(META_PATH)):
        return train_and_save()
    lr = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    pa = joblib.load(PA_MODEL_PATH) if os.path.exists(PA_MODEL_PATH) else None
    with open(META_PATH) as f:
        meta = json.load(f)
    return lr, vec, pa, meta


MODEL, VECTORIZER, PA_MODEL, META = load_models()


# ---------------------------------------------------------------------------
# Prediction core
# ---------------------------------------------------------------------------
def explain_prediction(text_clean: str, predicted_class: int, top_n: int = 6):
    """Return the tokens in the article that most pushed the model toward its decision."""
    feature_names = VECTORIZER.get_feature_names_out()
    vec = VECTORIZER.transform([text_clean])
    coefs = MODEL.coef_[0]  # positive => REAL, negative => FAKE
    nz = vec.nonzero()[1]
    contributions = []
    for idx in nz:
        weight = coefs[idx] * vec[0, idx]
        contributions.append((feature_names[idx], float(weight)))
    if predicted_class == 1:  # REAL — sort by most positive
        contributions.sort(key=lambda x: x[1], reverse=True)
    else:  # FAKE — sort by most negative
        contributions.sort(key=lambda x: x[1])
    return [w for w, _ in contributions[:top_n] if w]


def analyze(text: str):
    if not text or not text.strip():
        raise ValueError("Empty text provided")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text contains no usable words")

    vec = VECTORIZER.transform([cleaned])
    proba = MODEL.predict_proba(vec)[0]  # [P(FAKE), P(REAL)]
    pred = int(np.argmax(proba))
    confidence = float(proba[pred]) * 100
    label = "REAL" if pred == 1 else "FAKE"

    authenticity = float(proba[1]) * 100  # always probability of REAL
    if authenticity >= 75:
        risk = "Low"
    elif authenticity >= 45:
        risk = "Medium"
    else:
        risk = "High"

    sentiment_label, sentiment_val = sentiment_score(text)
    keywords = top_keywords(text)
    explanation_tokens = explain_prediction(cleaned, pred)

    # Trust score blends confidence + authenticity + length sanity
    length_factor = min(len(text.split()) / 60, 1.0)  # short snippets reduce trust
    trust_score = round((confidence * 0.5 + authenticity * 0.4 + length_factor * 10), 1)
    trust_score = float(min(max(trust_score, 0), 100))

    if label == "REAL":
        reasoning = (
            f"The article uses vocabulary, structure, and sourcing patterns consistent with "
            f"verified reporting. Key indicators: {', '.join(explanation_tokens[:4]) or 'balanced factual language'}."
        )
    else:
        reasoning = (
            f"The article shows linguistic patterns common in misleading or sensational content "
            f"(emotive wording, unverifiable claims). Key indicators: "
            f"{', '.join(explanation_tokens[:4]) or 'sensational phrasing'}."
        )

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "authenticity_score": round(authenticity, 2),
        "risk_level": risk,
        "trust_score": trust_score,
        "sentiment": {"label": sentiment_label, "score": sentiment_val},
        "keywords": keywords,
        "explanation": {
            "summary": reasoning,
            "top_signals": explanation_tokens,
        },
        "word_count": len(text.split()),
        "model": "TF-IDF + Logistic Regression",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def record_analytics(result):
    ANALYTICS["total"] += 1
    if result["prediction"] == "REAL":
        ANALYTICS["real"] += 1
    else:
        ANALYTICS["fake"] += 1
    c = result["confidence"]
    bucket = "90-100" if c >= 90 else "80-90" if c >= 80 else "70-80" if c >= 70 else "60-70" if c >= 60 else "50-60"
    ANALYTICS["confidence_buckets"][bucket] += 1
    ANALYTICS["recent"].insert(0, {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "risk": result["risk_level"],
        "timestamp": result["timestamp"],
    })
    ANALYTICS["recent"] = ANALYTICS["recent"][:10]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", meta=META)


@app.route("/predict", methods=["POST"])
def predict():
    text = (request.form.get("text") or request.json.get("text", "") if request.is_json else request.form.get("text", "")).strip()
    try:
        result = analyze(text)
        record_analytics(result)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    url = (request.form.get("url") or (request.json or {}).get("url", "")).strip()
    if not url:
        return jsonify({"success": False, "error": "URL is required"}), 400
    try:
        article = extract_article_from_url(url)
        if not article or len(article) < 80:
            return jsonify({"success": False, "error": "Could not extract sufficient article content from URL."}), 400
        result = analyze(article)
        result["source_url"] = url
        result["extracted_preview"] = article[:400] + ("..." if len(article) > 400 else "")
        record_analytics(result)
        return jsonify({"success": True, "result": result})
    except requests.RequestException as e:
        return jsonify({"success": False, "error": f"Failed to fetch URL: {e}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"success": False, "error": "Field 'text' is required"}), 400
    try:
        result = analyze(text)
        record_analytics(result)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/analytics")
def api_analytics():
    return jsonify({"success": True, "analytics": ANALYTICS, "model_meta": META})


@app.route("/api/contact", methods=["POST"])
def api_contact():
    data = request.get_json(silent=True) or request.form.to_dict()
    required = ["name", "email", "subject", "message"]
    if not all(data.get(k) for k in required):
        return jsonify({"success": False, "error": "All fields are required"}), 400
    # Persist or email integration would go here.
    return jsonify({"success": True, "message": "Message received. We'll be in touch."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

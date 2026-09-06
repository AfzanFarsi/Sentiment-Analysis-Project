"""
app.py - Sentiscope website backend (Flask)

Routes:
  GET  /                    -> welcome / home page
  GET  /dashboard           -> Power BI dashboard screenshots gallery
  GET  /predict             -> live sentiment reading demo
  GET  /about                -> methodology / about page
  POST /api/analyze         -> {text: "..."} -> sentiment prediction + word contributions
  GET  /api/stats           -> headline model/dataset stats

Run:
  pip install -r requirements.txt
  python app.py
Then open http://127.0.0.1:5000
"""

import os
import re
import json

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load the text-only model (see model/train_text_model.py)
# ---------------------------------------------------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "text_tfidf.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "text_sentiment_model.joblib"))

with open(os.path.join(MODEL_DIR, "text_model_metrics.json")) as f:
    MODEL_METRICS = json.load(f)

LABELS = ["negative", "neutral", "positive"]
FEATURE_NAMES = tfidf.get_feature_names_out()

# Same cleaning logic used to train the model (kept in sync with
# scripts/01_preprocessing.py / run_pipeline.py)
STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for with
at by from up down out over under again further then once here there when where
why how all any both each few more most other some such no nor not only own same
so than too very s t can will just don should now i me my myself we our ours
ourselves you your yours yourself yourselves he him his himself she her hers
herself it its itself they them their theirs themselves what which who whom this
that these those am do does did having have has had having im ive youre youve
u ur going gonna wanna gotta got get getting im ill
""".split())

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_SYMBOL_RE.sub("", t)
    t = NON_ALPHA_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


@app.route("/")
def home():
    return render_template("home.html", metrics=MODEL_METRICS, active_page="home")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=MODEL_METRICS, active_page="dashboard")


@app.route("/predict")
def predict():
    return render_template("predict.html", metrics=MODEL_METRICS, active_page="predict")


@app.route("/about")
def about():
    return render_template("about.html", metrics=MODEL_METRICS, active_page="about")


@app.route("/api/stats")
def stats():
    return jsonify({
        "accuracy": round(MODEL_METRICS["accuracy"] * 100, 1),
        "macro_f1": round(MODEL_METRICS["macro_f1"] * 100, 1),
        "train_size": MODEL_METRICS["train_size"],
        "test_size": MODEL_METRICS["test_size"],
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    raw_text = (payload.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(raw_text) > 2000:
        return jsonify({"error": "Please keep text under 2000 characters."}), 400

    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify({"error": "That text doesn't contain enough recognizable words to analyze."}), 400

    vec = tfidf.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = LABELS[pred_idx]

    probabilities = {LABELS[i]: round(float(proba[i]), 4) for i in range(len(LABELS))}

    # ---- word-level contribution toward the predicted class ----
    coefs = model.coef_[pred_idx]
    nonzero = vec.nonzero()[1]
    contributions = []
    for idx in nonzero:
        weight = float(coefs[idx] * vec[0, idx])
        contributions.append({"word": FEATURE_NAMES[idx], "weight": weight})
    contributions.sort(key=lambda x: x["weight"], reverse=True)
    top_contributions = [c for c in contributions if c["weight"] > 0][:8]

    return jsonify({
        "sentiment": pred_label,
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": probabilities,
        "top_words": top_contributions,
        "cleaned_text": cleaned,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

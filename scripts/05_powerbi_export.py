"""
05_powerbi_export.py
Visualize in Power BI - Sentiment Analysis Project

Exports flat, Power-BI-friendly CSV tables:
1. predictions.csv          - row-level: text, actual/predicted sentiment, correctness, confidence, demographics
2. sentiment_by_time.csv    - sentiment counts/proportions by Time of Tweet (trend over the day)
3. sentiment_by_age.csv     - sentiment counts/proportions by Age of User group
4. sentiment_by_country.csv - sentiment counts/proportions by Country (top 30)
5. top_words_by_sentiment.csv - word impact table (word, sentiment, weight) for word-cloud/impact visuals
6. model_performance.csv    - metrics per model for a KPI card / comparison visual
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

OUT = "../powerbi_exports"
LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

test_df = pd.read_csv("../data/processed/test_clean.csv")
X_test_full = sp.load_npz("../data/processed/X_test_full.npz")
y_test = np.load("../data/processed/y_test.npy")

model = joblib.load("../models/best_sentiment_model.joblib")
pred = model.predict(X_test_full)
proba = model.predict_proba(X_test_full)
confidence = proba.max(axis=1)

# ---------- 1. Row-level predictions table ----------
pred_df = pd.DataFrame({
    "textID": test_df["textID"],
    "text": test_df["text"],
    "actual_sentiment": test_df["sentiment"],
    "predicted_sentiment": [LABEL_MAP[p] for p in pred],
    "confidence": np.round(confidence, 4),
    "is_correct": (test_df["sentiment"].values == np.array([LABEL_MAP[p] for p in pred])),
    "time_of_tweet": test_df["Time of Tweet"],
    "age_group": test_df["Age of User"],
    "country": test_df["Country"],
    "word_count": test_df["word_count"],
    "text_length": test_df["text_length"],
})
pred_df.to_csv(f"{OUT}/predictions.csv", index=False)

# ---------- 2. Sentiment by Time of Tweet ----------
time_order = ["morning", "noon", "night"]
by_time = (pred_df.groupby(["time_of_tweet", "predicted_sentiment"])
           .size().reset_index(name="count"))
by_time["proportion"] = by_time.groupby("time_of_tweet")["count"].transform(lambda x: x / x.sum())
by_time["time_of_tweet"] = pd.Categorical(by_time["time_of_tweet"], categories=time_order, ordered=True)
by_time = by_time.sort_values("time_of_tweet")
by_time.to_csv(f"{OUT}/sentiment_by_time.csv", index=False)

# ---------- 3. Sentiment by Age Group ----------
age_order = ["0-20", "21-30", "31-45", "46-60", "60-70", "70-100"]
by_age = (pred_df.groupby(["age_group", "predicted_sentiment"])
          .size().reset_index(name="count"))
by_age["proportion"] = by_age.groupby("age_group")["count"].transform(lambda x: x / x.sum())
by_age["age_group"] = pd.Categorical(by_age["age_group"], categories=age_order, ordered=True)
by_age = by_age.sort_values("age_group")
by_age.to_csv(f"{OUT}/sentiment_by_age.csv", index=False)

# ---------- 4. Sentiment by Country (top 30 by volume) ----------
top_countries = pred_df["country"].value_counts().head(30).index
by_country = (pred_df[pred_df["country"].isin(top_countries)]
              .groupby(["country", "predicted_sentiment"]).size().reset_index(name="count"))
by_country["proportion"] = by_country.groupby("country")["count"].transform(lambda x: x / x.sum())
by_country.to_csv(f"{OUT}/sentiment_by_country.csv", index=False)

# ---------- 5. Top words / word impact table ----------
tfidf = joblib.load("../models/tfidf_vectorizer.joblib")
feature_names = tfidf.get_feature_names_out()
train_df = pd.read_csv("../data/processed/train_clean.csv")
X_train_tfidf = tfidf.transform(train_df["clean_text"])

rows = []
for label_idx, label_name in LABEL_MAP.items():
    mask = (train_df["sentiment_label"] == label_idx).values
    mean_tfidf = np.asarray(X_train_tfidf[mask].mean(axis=0)).ravel()
    top_idx = np.argsort(mean_tfidf)[-40:][::-1]
    for i in top_idx:
        rows.append({"sentiment": label_name, "word": feature_names[i],
                      "weight": round(float(mean_tfidf[i]), 5)})
word_impact_df = pd.DataFrame(rows)
word_impact_df.to_csv(f"{OUT}/top_words_by_sentiment.csv", index=False)

# ---------- 6. Model performance summary ----------
with open("../models/metrics_summary.json") as f:
    metrics = json.load(f)

perf_rows = []
for model_name, m in metrics["validation_results_all_models"].items():
    perf_rows.append({
        "model": model_name,
        "dataset": "validation",
        "accuracy": round(m["val_accuracy"], 4),
        "precision_macro": round(m["val_precision_macro"], 4),
        "recall_macro": round(m["val_recall_macro"], 4),
        "f1_macro": round(m["val_f1_macro"], 4),
        "is_best_model": model_name == metrics["best_model"],
    })
perf_rows.append({
    "model": metrics["best_model"],
    "dataset": "test (holdout)",
    "accuracy": round(metrics["test_accuracy"], 4),
    "precision_macro": round(metrics["test_macro_precision"], 4),
    "recall_macro": round(metrics["test_macro_recall"], 4),
    "f1_macro": round(metrics["test_macro_f1"], 4),
    "is_best_model": True,
})
perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(f"{OUT}/model_performance.csv", index=False)

print("Power BI export complete. Files written to ../powerbi_exports/:")
for f in ["predictions.csv", "sentiment_by_time.csv", "sentiment_by_age.csv",
          "sentiment_by_country.csv", "top_words_by_sentiment.csv", "model_performance.csv"]:
    print(f" - {f}")

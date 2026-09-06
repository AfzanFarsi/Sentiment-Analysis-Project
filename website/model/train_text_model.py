"""
train_text_model.py
Trains a TEXT-ONLY sentiment classifier for the website's live demo.

The main project model (in ../models/) uses TF-IDF + demographic features
(time of day, age, country, population stats). That's great for the batch
analysis / Power BI pipeline, but a website visitor typing a sentence has
no demographic data to give us. So this script trains a second, lighter
model using ONLY the tweet text -> sentiment, which is what a live text
box can actually support.

Run once from this folder:  python train_text_model.py
Outputs: text_tfidf.joblib, text_sentiment_model.joblib, text_model_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

LABELS = ["negative", "neutral", "positive"]

train_df = pd.read_csv(os.path.join(PROC_DIR, "train_clean.csv"))
test_df = pd.read_csv(os.path.join(PROC_DIR, "test_clean.csv"))

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
X_train = tfidf.fit_transform(train_df["clean_text"])
X_test = tfidf.transform(test_df["clean_text"])
y_train = train_df["sentiment_label"].values
y_test = test_df["sentiment_label"].values

model = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro")
report = classification_report(y_test, pred, target_names=LABELS, output_dict=True)

print(f"Text-only model  ->  accuracy={acc:.4f}  macro_f1={f1:.4f}")
print(classification_report(y_test, pred, target_names=LABELS))

joblib.dump(tfidf, os.path.join(BASE_DIR, "text_tfidf.joblib"))
joblib.dump(model, os.path.join(BASE_DIR, "text_sentiment_model.joblib"))

with open(os.path.join(BASE_DIR, "text_model_metrics.json"), "w") as f:
    json.dump({
        "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "classification_report": report,
        "train_size": len(train_df), "test_size": len(test_df),
    }, f, indent=2)

print("Saved text_tfidf.joblib, text_sentiment_model.joblib, text_model_metrics.json")

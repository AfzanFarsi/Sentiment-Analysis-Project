"""
run_pipeline.py
Sentiment Analysis Using Machine Learning - Project 11
SINGLE-FILE PIPELINE (combines preprocessing -> EDA -> feature engineering ->
modeling -> Power BI export into one script you can run top to bottom)

Usage:
    python run_pipeline.py

Requires: pip install -r requirements.txt
Expects raw data at: data/raw/train.csv and data/raw/test.csv
(paths are resolved relative to this file's location, so it works
regardless of which folder you run it from)
"""

import os
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from scipy.sparse import hstack, csr_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

# ======================================================================
# 0. PATHS & SETUP
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visuals")
PBI_DIR = os.path.join(BASE_DIR, "powerbi_exports")

for d in [PROC_DIR, MODEL_DIR, VIS_DIR, PBI_DIR]:
    os.makedirs(d, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
sns.set_theme(style="whitegrid")
PALETTE = {"negative": "#E4572E", "neutral": "#8C8C8C", "positive": "#2E8B57"}

print("=" * 70)
print("SENTIMENT ANALYSIS PIPELINE - STARTING")
print("=" * 70)

# ======================================================================
# STEP 1: PREPROCESS DATA
# ======================================================================
print("\n[1/5] Preprocessing data...")

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


def clean_text(text):
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


def load_and_clean(path, is_train):
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)
    removed = before - len(df)

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment_label"] = df["sentiment"].map(sentiment_map)

    df["Time of Tweet"] = df["Time of Tweet"].str.strip().str.lower()
    df["Age of User"] = df["Age of User"].str.strip()
    df["Country"] = df["Country"].str.strip()

    for col in ["Population -2020", "Land Area (Km²)", "Density (P/Km²)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["clean_text"].str.split().apply(len)

    print(f"  {'TRAIN' if is_train else 'TEST'}: {before} rows -> "
          f"removed {removed} dupes -> final {df.shape}")
    return df


train_df = load_and_clean(os.path.join(RAW_DIR, "train.csv"), True)
test_df = load_and_clean(os.path.join(RAW_DIR, "test.csv"), False)

train_df.to_csv(os.path.join(PROC_DIR, "train_clean.csv"), index=False)
test_df.to_csv(os.path.join(PROC_DIR, "test_clean.csv"), index=False)

# ======================================================================
# STEP 2: EDA
# ======================================================================
print("\n[2/5] Running EDA and saving charts...")

order = LABELS

plt.figure(figsize=(6, 5))
counts = train_df["sentiment"].value_counts().reindex(order)
bars = plt.bar(order, counts.values, color=[PALETTE[o] for o in order])
plt.title("Sentiment Class Distribution", fontsize=14, fontweight="bold")
plt.ylabel("Number of Tweets")
for b, v in zip(bars, counts.values):
    plt.text(b.get_x() + b.get_width() / 2, v + 100, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "01_sentiment_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
for s in order:
    sns.kdeplot(train_df[train_df["sentiment"] == s]["word_count"],
                label=s, color=PALETTE[s], fill=True, alpha=0.3)
plt.title("Word Count Distribution by Sentiment", fontsize=14, fontweight="bold")
plt.xlabel("Word Count (cleaned text)")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "02_wordcount_distribution.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, s in zip(axes, order):
    words = " ".join(train_df[train_df["sentiment"] == s]["clean_text"].astype(str)).split()
    top = Counter(words).most_common(12)
    words_, freqs_ = zip(*top)
    ax.barh(words_[::-1], freqs_[::-1], color=PALETTE[s])
    ax.set_title(f"Top Words: {s.capitalize()}", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "03_top_words_by_sentiment.png"), dpi=150)
plt.close()

ct = pd.crosstab(train_df["Time of Tweet"], train_df["sentiment"], normalize="index")[order]
ct = ct.reindex(["morning", "noon", "night"])
ct.plot(kind="bar", stacked=True, color=[PALETTE[o] for o in order], figsize=(7, 5))
plt.title("Sentiment Proportion by Time of Tweet", fontsize=14, fontweight="bold")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "04_sentiment_by_time.png"), dpi=150)
plt.close()

age_order = ["0-20", "21-30", "31-45", "46-60", "60-70", "70-100"]
ct2 = pd.crosstab(train_df["Age of User"], train_df["sentiment"], normalize="index").reindex(age_order)[order]
ct2.plot(kind="bar", stacked=True, color=[PALETTE[o] for o in order], figsize=(8, 5))
plt.title("Sentiment Proportion by Age Group", fontsize=14, fontweight="bold")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "05_sentiment_by_age.png"), dpi=150)
plt.close()

top_countries = train_df["Country"].value_counts().head(15).index
sub = train_df[train_df["Country"].isin(top_countries)]
ct3 = pd.crosstab(sub["Country"], sub["sentiment"], normalize="index").reindex(top_countries)[order]
ct3.plot(kind="barh", stacked=True, color=[PALETTE[o] for o in order], figsize=(8, 7))
plt.title("Sentiment Mix - Top 15 Countries by Volume", fontsize=14, fontweight="bold")
plt.xlabel("Proportion")
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "06_sentiment_by_country.png"), dpi=150)
plt.close()

numeric_cols = ["sentiment_label", "text_length", "word_count",
                 "Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
plt.figure(figsize=(7, 6))
corr = train_df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap - Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "07_correlation_heatmap.png"), dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=train_df, x="sentiment", y="text_length", order=order, hue="sentiment",
            palette=PALETTE, legend=False)
plt.title("Text Length Outliers by Sentiment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "08_textlength_boxplot.png"), dpi=150)
plt.close()

print("  8 EDA charts saved to visuals/")

# ======================================================================
# STEP 3: ENGINEER FEATURES
# ======================================================================
print("\n[3/5] Engineering features (TF-IDF, encoding, scaling)...")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
X_test_tfidf = tfidf.transform(test_df["clean_text"])

svd = TruncatedSVD(n_components=100, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)
print(f"  TruncatedSVD explained variance (100 comps): {svd.explained_variance_ratio_.sum():.3f}")

cat_cols = ["Time of Tweet", "Age of User"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_cat = ohe.fit_transform(train_df[cat_cols])
X_test_cat = ohe.transform(test_df[cat_cols])

country_freq = train_df["Country"].value_counts(normalize=True)
train_df["country_freq"] = train_df["Country"].map(country_freq).fillna(0)
test_df["country_freq"] = test_df["Country"].map(country_freq).fillna(0)

num_cols = ["text_length", "word_count", "country_freq",
            "Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(train_df[num_cols])
X_test_num = scaler.transform(test_df[num_cols])

X_train_full = hstack([X_train_tfidf, X_train_cat, csr_matrix(X_train_num)]).tocsr()
X_test_full = hstack([X_test_tfidf, X_test_cat, csr_matrix(X_test_num)]).tocsr()

y_train = train_df["sentiment_label"].values
y_test = test_df["sentiment_label"].values

print(f"  Final training matrix: {X_train_full.shape} | test matrix: {X_test_full.shape}")

joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(svd, os.path.join(MODEL_DIR, "svd_transformer.joblib"))
joblib.dump(ohe, os.path.join(MODEL_DIR, "onehot_encoder.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(country_freq, os.path.join(MODEL_DIR, "country_freq_map.joblib"))

sp.save_npz(os.path.join(PROC_DIR, "X_train_full.npz"), X_train_full)
sp.save_npz(os.path.join(PROC_DIR, "X_test_full.npz"), X_test_full)
np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROC_DIR, "y_test.npy"), y_test)
np.save(os.path.join(PROC_DIR, "X_train_svd.npy"), X_train_svd)
np.save(os.path.join(PROC_DIR, "X_test_svd.npy"), X_test_svd)

# ======================================================================
# STEP 4: BUILD ML MODEL
# ======================================================================
print("\n[4/5] Training and evaluating models...")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.15, random_state=42, stratify=y_train
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1),
    "Multinomial Naive Bayes": MultinomialNB(),
}

results = {}
fitted_models = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_val, val_pred, average="macro")
    results[name] = {"val_accuracy": acc, "val_precision_macro": p,
                      "val_recall_macro": r, "val_f1_macro": f1}
    fitted_models[name] = model
    print(f"  {name}: val_acc={acc:.4f}  macro_f1={f1:.4f}")

best_name = max(results, key=lambda k: results[k]["val_f1_macro"])
best_model = fitted_models[best_name]
print(f"  Best model: {best_name}")

best_model.fit(X_train_full, y_train)

test_pred = best_model.predict(X_test_full)
test_acc = accuracy_score(y_test, test_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="macro")
report = classification_report(y_test, test_pred, target_names=LABELS, output_dict=True)
print(f"  TEST -> accuracy={test_acc:.4f}  macro_f1={f1:.4f}")
print(classification_report(y_test, test_pred, target_names=LABELS))

cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title(f"Confusion Matrix - {best_name} (Test Set)", fontsize=13, fontweight="bold")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "09_confusion_matrix.png"), dpi=150)
plt.close()

comp_df = pd.DataFrame(results).T
comp_df[["val_accuracy", "val_f1_macro"]].plot(kind="bar", figsize=(8, 5),
                                                  color=["#4C72B0", "#DD8452"])
plt.title("Model Comparison (Validation Set)", fontsize=13, fontweight="bold")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(["Accuracy", "Macro F1"])
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "10_model_comparison.png"), dpi=150)
plt.close()

feature_names = tfidf.get_feature_names_out()
if best_name == "Logistic Regression":
    coefs = best_model.coef_
    n_tfidf = len(feature_names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for i, (ax, label) in enumerate(zip(axes, LABELS)):
        class_coefs = coefs[i][:n_tfidf]
        top_idx = np.argsort(class_coefs)[-12:]
        ax.barh([feature_names[j] for j in top_idx], class_coefs[top_idx], color="#2E8B57")
        ax.set_title(f"Top Words Driving: {label.capitalize()}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "11_feature_impact.png"), dpi=150)
    plt.close()

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_sentiment_model.joblib"))

metrics_summary = {
    "best_model": best_name,
    "validation_results_all_models": results,
    "test_accuracy": test_acc,
    "test_macro_precision": p,
    "test_macro_recall": r,
    "test_macro_f1": f1,
    "test_classification_report": report,
}
with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

# ======================================================================
# STEP 5: VISUALIZE IN POWER BI (export flat CSVs)
# ======================================================================
print("\n[5/5] Exporting Power BI-ready CSVs...")

proba = best_model.predict_proba(X_test_full)
confidence = proba.max(axis=1)

pred_df = pd.DataFrame({
    "textID": test_df["textID"],
    "text": test_df["text"],
    "actual_sentiment": test_df["sentiment"],
    "predicted_sentiment": [LABEL_MAP[p_] for p_ in test_pred],
    "confidence": np.round(confidence, 4),
    "is_correct": (test_df["sentiment"].values == np.array([LABEL_MAP[p_] for p_ in test_pred])),
    "time_of_tweet": test_df["Time of Tweet"],
    "age_group": test_df["Age of User"],
    "country": test_df["Country"],
    "word_count": test_df["word_count"],
    "text_length": test_df["text_length"],
})
pred_df.to_csv(os.path.join(PBI_DIR, "predictions.csv"), index=False)

time_order = ["morning", "noon", "night"]
by_time = pred_df.groupby(["time_of_tweet", "predicted_sentiment"]).size().reset_index(name="count")
by_time["proportion"] = by_time.groupby("time_of_tweet")["count"].transform(lambda x: x / x.sum())
by_time["time_of_tweet"] = pd.Categorical(by_time["time_of_tweet"], categories=time_order, ordered=True)
by_time = by_time.sort_values("time_of_tweet")
by_time.to_csv(os.path.join(PBI_DIR, "sentiment_by_time.csv"), index=False)

by_age = pred_df.groupby(["age_group", "predicted_sentiment"]).size().reset_index(name="count")
by_age["proportion"] = by_age.groupby("age_group")["count"].transform(lambda x: x / x.sum())
by_age["age_group"] = pd.Categorical(by_age["age_group"], categories=age_order, ordered=True)
by_age = by_age.sort_values("age_group")
by_age.to_csv(os.path.join(PBI_DIR, "sentiment_by_age.csv"), index=False)

top_countries_pbi = pred_df["country"].value_counts().head(30).index
by_country = (pred_df[pred_df["country"].isin(top_countries_pbi)]
              .groupby(["country", "predicted_sentiment"]).size().reset_index(name="count"))
by_country["proportion"] = by_country.groupby("country")["count"].transform(lambda x: x / x.sum())
by_country.to_csv(os.path.join(PBI_DIR, "sentiment_by_country.csv"), index=False)

rows = []
for label_idx, label_name in LABEL_MAP.items():
    mask = (train_df["sentiment_label"] == label_idx).values
    mean_tfidf = np.asarray(X_train_tfidf[mask].mean(axis=0)).ravel()
    top_idx = np.argsort(mean_tfidf)[-40:][::-1]
    for i in top_idx:
        rows.append({"sentiment": label_name, "word": feature_names[i],
                      "weight": round(float(mean_tfidf[i]), 5)})
pd.DataFrame(rows).to_csv(os.path.join(PBI_DIR, "top_words_by_sentiment.csv"), index=False)

perf_rows = []
for model_name, m in results.items():
    perf_rows.append({
        "model": model_name, "dataset": "validation",
        "accuracy": round(m["val_accuracy"], 4),
        "precision_macro": round(m["val_precision_macro"], 4),
        "recall_macro": round(m["val_recall_macro"], 4),
        "f1_macro": round(m["val_f1_macro"], 4),
        "is_best_model": model_name == best_name,
    })
perf_rows.append({
    "model": best_name, "dataset": "test (holdout)",
    "accuracy": round(test_acc, 4), "precision_macro": round(p, 4),
    "recall_macro": round(r, 4), "f1_macro": round(f1, 4), "is_best_model": True,
})
pd.DataFrame(perf_rows).to_csv(os.path.join(PBI_DIR, "model_performance.csv"), index=False)

print("  6 CSVs saved to powerbi_exports/")

print("\n" + "=" * 70)
print(f"PIPELINE COMPLETE — best model: {best_name} | test accuracy: {test_acc:.2%}")
print("=" * 70)

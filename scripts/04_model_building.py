"""
04_model_building.py
Build ML Model - Sentiment Analysis Project

- Load engineered features (train already split by original dataset design;
  test.csv acts as the held-out evaluation set)
- Train Logistic Regression, Random Forest, and Multinomial Naive Bayes
- Evaluate with accuracy, precision, recall, F1 (macro + per-class), confusion matrix
- Select best model, save it, and generate a metrics report
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

LABELS = ["negative", "neutral", "positive"]

X_train_full = sp.load_npz("../data/processed/X_train_full.npz")
X_test_full = sp.load_npz("../data/processed/X_test_full.npz")
y_train = np.load("../data/processed/y_train.npy")
y_test = np.load("../data/processed/y_test.npy")

# Carve a validation split out of train for model selection
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
    print(f"{name}: val_acc={acc:.4f}  macro_f1={f1:.4f}")

# ---------- Select best model by validation macro F1 ----------
best_name = max(results, key=lambda k: results[k]["val_f1_macro"])
best_model = fitted_models[best_name]
print(f"\nBest model: {best_name}")

# Refit best model on FULL training data
best_model.fit(X_train_full, y_train)

# ---------- Evaluate on held-out test.csv ----------
test_pred = best_model.predict(X_test_full)
test_proba = best_model.predict_proba(X_test_full) if hasattr(best_model, "predict_proba") else None

test_acc = accuracy_score(y_test, test_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="macro")
report = classification_report(y_test, test_pred, target_names=LABELS, output_dict=True)

print(f"\nTEST SET  -> accuracy={test_acc:.4f}  macro_f1={f1:.4f}")
print(classification_report(y_test, test_pred, target_names=LABELS))

# ---------- Confusion matrix plot ----------
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title(f"Confusion Matrix - {best_name} (Test Set)", fontsize=13, fontweight="bold")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("../visuals/09_confusion_matrix.png", dpi=150)
plt.close()

# ---------- Model comparison bar chart ----------
comp_df = pd.DataFrame(results).T
plt.figure(figsize=(8, 5))
comp_df[["val_accuracy", "val_f1_macro"]].plot(kind="bar", figsize=(8, 5),
                                                  color=["#4C72B0", "#DD8452"])
plt.title("Model Comparison (Validation Set)", fontsize=13, fontweight="bold")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(["Accuracy", "Macro F1"])
plt.tight_layout()
plt.savefig("../visuals/10_model_comparison.png", dpi=150)
plt.close()

# ---------- Feature importance (top TF-IDF terms) for interpretability ----------
tfidf = joblib.load("../models/tfidf_vectorizer.joblib")
feature_names = tfidf.get_feature_names_out()

if best_name == "Logistic Regression":
    coefs = best_model.coef_  # shape (n_classes, n_features_total) but includes non-tfidf feats
    n_tfidf = len(feature_names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for i, (ax, label) in enumerate(zip(axes, LABELS)):
        class_coefs = coefs[i][:n_tfidf]
        top_idx = np.argsort(class_coefs)[-12:]
        ax.barh([feature_names[j] for j in top_idx], class_coefs[top_idx], color="#2E8B57")
        ax.set_title(f"Top Words Driving: {label.capitalize()}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("../visuals/11_feature_impact.png", dpi=150)
    plt.close()

# ---------- Save best model + metrics ----------
joblib.dump(best_model, "../models/best_sentiment_model.joblib")

metrics_summary = {
    "best_model": best_name,
    "validation_results_all_models": results,
    "test_accuracy": test_acc,
    "test_macro_precision": p,
    "test_macro_recall": r,
    "test_macro_f1": f1,
    "test_classification_report": report,
}
with open("../models/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("\nSaved best model, confusion matrix, comparison chart, and metrics_summary.json")

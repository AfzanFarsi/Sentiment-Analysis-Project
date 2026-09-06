"""
03_feature_engineering.py
Engineer Features - Sentiment Analysis Project

- TF-IDF vectorization of cleaned tweet text (uni+bi-grams)
- One-hot encode demographic categoricals (Time of Tweet, Age of User)
- Target-encode / frequency-encode high-cardinality Country column
- Combine numeric + text features
- Dimensionality reduction on TF-IDF via TruncatedSVD (for a compact dense feature set)
- Save fitted vectorizer/encoders + transformed feature matrices for modeling
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack, csr_matrix

train_df = pd.read_csv("../data/processed/train_clean.csv")
test_df = pd.read_csv("../data/processed/test_clean.csv")

# ---------- TF-IDF on cleaned text ----------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
X_test_tfidf = tfidf.transform(test_df["clean_text"])

# ---------- Dimensionality reduction (for viz / lighter models) ----------
svd = TruncatedSVD(n_components=100, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)
print(f"TruncatedSVD explained variance (100 comps): {svd.explained_variance_ratio_.sum():.3f}")

# ---------- One-hot encode categorical demographic features ----------
cat_cols = ["Time of Tweet", "Age of User"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_cat = ohe.fit_transform(train_df[cat_cols])
X_test_cat = ohe.transform(test_df[cat_cols])

# ---------- Frequency-encode Country (high cardinality) ----------
country_freq = train_df["Country"].value_counts(normalize=True)
train_df["country_freq"] = train_df["Country"].map(country_freq).fillna(0)
test_df["country_freq"] = test_df["Country"].map(country_freq).fillna(0)

# ---------- Scale numeric features (MinMax keeps values non-negative so
# the combined matrix stays compatible with MultinomialNB) ----------
num_cols = ["text_length", "word_count", "country_freq",
            "Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(train_df[num_cols])
X_test_num = scaler.transform(test_df[num_cols])

# ---------- Combine all features into one sparse matrix (for modeling) ----------
X_train_full = hstack([X_train_tfidf, X_train_cat, csr_matrix(X_train_num)]).tocsr()
X_test_full = hstack([X_test_tfidf, X_test_cat, csr_matrix(X_test_num)]).tocsr()

y_train = train_df["sentiment_label"].values
y_test = test_df["sentiment_label"].values

print(f"Final training feature matrix: {X_train_full.shape}")
print(f"Final test feature matrix: {X_test_full.shape}")

# ---------- Persist everything needed downstream ----------
joblib.dump(tfidf, "../models/tfidf_vectorizer.joblib")
joblib.dump(svd, "../models/svd_transformer.joblib")
joblib.dump(ohe, "../models/onehot_encoder.joblib")
joblib.dump(scaler, "../models/scaler.joblib")
joblib.dump(country_freq, "../models/country_freq_map.joblib")

import scipy.sparse as sp
sp.save_npz("../data/processed/X_train_full.npz", X_train_full)
sp.save_npz("../data/processed/X_test_full.npz", X_test_full)
np.save("../data/processed/y_train.npy", y_train)
np.save("../data/processed/y_test.npy", y_test)
np.save("../data/processed/X_train_svd.npy", X_train_svd)
np.save("../data/processed/X_test_svd.npy", X_test_svd)

print("\nFeature engineering complete. Artifacts saved to ../models/ and ../data/processed/")

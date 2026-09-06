"""
01_preprocessing.py
Sentiment Analysis Using Machine Learning - Project 11

Step: Preprocess Data
- Load raw train/test CSVs (Twitter Sentiment Extraction dataset + demographic fields)
- Handle missing values
- Clean text (lowercase, remove URLs/mentions/punctuation/digits/extra whitespace)
- Remove stopwords (lightweight built-in list, no internet-dependent downloads)
- Remove duplicates
- Encode sentiment labels (negative=0, neutral=1, positive=2)
- Encode/clean categorical demographic columns
- Save cleaned data to data/processed/
"""

import re
import pandas as pd
import numpy as np

RAW_DIR = "../data/raw"
OUT_DIR = "../data/processed"

# A compact general-purpose English stopword list (avoids needing nltk downloads)
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
    t = HASHTAG_SYMBOL_RE.sub("", t)      # keep the hashtag word, drop the symbol
    t = NON_ALPHA_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def load_and_clean(path: str, is_train: bool) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # --- Handle missing values ---
    df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    # --- Remove exact duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)
    removed = before - len(df)

    # --- Clean text ---
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    # --- Encode sentiment ---
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment_label"] = df["sentiment"].map(sentiment_map)

    # --- Clean categorical/demographic columns ---
    df["Time of Tweet"] = df["Time of Tweet"].str.strip().str.lower()
    df["Age of User"] = df["Age of User"].str.strip()
    df["Country"] = df["Country"].str.strip()

    # --- Basic outlier handling on numeric population/density fields ---
    for col in ["Population -2020", "Land Area (Km²)", "Density (P/Km²)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            # cap extreme outliers at 1st/99th percentile
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

    # --- Feature: text length / word count ---
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["clean_text"].str.split().apply(len)

    print(f"{'TRAIN' if is_train else 'TEST'}: loaded {before} rows, "
          f"removed {removed} duplicates, final shape {df.shape}")

    return df


if __name__ == "__main__":
    train_df = load_and_clean(f"{RAW_DIR}/train.csv", is_train=True)
    test_df = load_and_clean(f"{RAW_DIR}/test.csv", is_train=False)

    train_df.to_csv(f"{OUT_DIR}/train_clean.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test_clean.csv", index=False)

    print("\nSaved cleaned files to data/processed/")
    print("\nTrain sentiment distribution:")
    print(train_df["sentiment"].value_counts())
    print("\nTest sentiment distribution:")
    print(test_df["sentiment"].value_counts())

"""
02_eda.py
Perform EDA - Sentiment Analysis Project

Generates visualizations saved to ../visuals/:
- Sentiment class distribution
- Text length / word count distributions by sentiment
- Top words per sentiment class (bar charts, TF-based, no external wordcloud dep)
- Sentiment by Time of Tweet, Age of User, Country (top 15)
- Correlation heatmap of numeric features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_theme(style="whitegrid")
PALETTE = {"negative": "#E4572E", "neutral": "#8C8C8C", "positive": "#2E8B57"}

df = pd.read_csv("../data/processed/train_clean.csv")

# ---------- 1. Sentiment distribution ----------
plt.figure(figsize=(6, 5))
order = ["negative", "neutral", "positive"]
counts = df["sentiment"].value_counts().reindex(order)
bars = plt.bar(order, counts.values, color=[PALETTE[o] for o in order])
plt.title("Sentiment Class Distribution", fontsize=14, fontweight="bold")
plt.ylabel("Number of Tweets")
for b, v in zip(bars, counts.values):
    plt.text(b.get_x() + b.get_width()/2, v + 100, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("../visuals/01_sentiment_distribution.png", dpi=150)
plt.close()

# ---------- 2. Text length distribution by sentiment ----------
plt.figure(figsize=(8, 5))
for s in order:
    subset = df[df["sentiment"] == s]["word_count"]
    sns.kdeplot(subset, label=s, color=PALETTE[s], fill=True, alpha=0.3)
plt.title("Word Count Distribution by Sentiment", fontsize=14, fontweight="bold")
plt.xlabel("Word Count (cleaned text)")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("../visuals/02_wordcount_distribution.png", dpi=150)
plt.close()

# ---------- 3. Top words per sentiment ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, s in zip(axes, order):
    text_blob = " ".join(df[df["sentiment"] == s]["clean_text"].astype(str))
    words = text_blob.split()
    top = Counter(words).most_common(12)
    words_, freqs_ = zip(*top)
    ax.barh(words_[::-1], freqs_[::-1], color=PALETTE[s])
    ax.set_title(f"Top Words: {s.capitalize()}", fontweight="bold")
plt.tight_layout()
plt.savefig("../visuals/03_top_words_by_sentiment.png", dpi=150)
plt.close()

# ---------- 4. Sentiment by Time of Tweet ----------
plt.figure(figsize=(7, 5))
ct = pd.crosstab(df["Time of Tweet"], df["sentiment"], normalize="index")[order]
ct = ct.reindex(["morning", "noon", "night"])
ct.plot(kind="bar", stacked=True, color=[PALETTE[o] for o in order], figsize=(7, 5))
plt.title("Sentiment Proportion by Time of Tweet", fontsize=14, fontweight="bold")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("../visuals/04_sentiment_by_time.png", dpi=150)
plt.close()

# ---------- 5. Sentiment by Age Group ----------
age_order = ["0-20", "21-30", "31-45", "46-60", "60-70", "70-100"]
plt.figure(figsize=(8, 5))
ct2 = pd.crosstab(df["Age of User"], df["sentiment"], normalize="index").reindex(age_order)[order]
ct2.plot(kind="bar", stacked=True, color=[PALETTE[o] for o in order], figsize=(8, 5))
plt.title("Sentiment Proportion by Age Group", fontsize=14, fontweight="bold")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("../visuals/05_sentiment_by_age.png", dpi=150)
plt.close()

# ---------- 6. Top 15 countries by tweet volume + sentiment mix ----------
top_countries = df["Country"].value_counts().head(15).index
sub = df[df["Country"].isin(top_countries)]
ct3 = pd.crosstab(sub["Country"], sub["sentiment"], normalize="index").reindex(top_countries)[order]
ct3.plot(kind="barh", stacked=True, color=[PALETTE[o] for o in order], figsize=(8, 7))
plt.title("Sentiment Mix - Top 15 Countries by Volume", fontsize=14, fontweight="bold")
plt.xlabel("Proportion")
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("../visuals/06_sentiment_by_country.png", dpi=150)
plt.close()

# ---------- 7. Correlation heatmap of numeric features ----------
numeric_cols = ["sentiment_label", "text_length", "word_count",
                 "Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
plt.figure(figsize=(7, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap - Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../visuals/07_correlation_heatmap.png", dpi=150)
plt.close()

# ---------- 8. Boxplot: text length outliers by sentiment ----------
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="sentiment", y="text_length", order=order,
            palette=PALETTE)
plt.title("Text Length Outliers by Sentiment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../visuals/08_textlength_boxplot.png", dpi=150)
plt.close()

print("EDA complete. 8 visualizations saved to ../visuals/")
print("\nNumeric feature correlation with sentiment_label:")
print(corr["sentiment_label"].sort_values(ascending=False))

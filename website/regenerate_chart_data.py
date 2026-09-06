"""
regenerate_chart_data.py
Re-exports ../powerbi_exports/*.csv into static/data/*.json for the dashboard
charts. Run this after re-running the main pipeline if you want the website's
dashboard to reflect updated numbers.

Usage (from the website/ folder):
    python regenerate_chart_data.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PBI = os.path.join(BASE_DIR, "..", "powerbi_exports")
OUT = os.path.join(BASE_DIR, "static", "data")
os.makedirs(OUT, exist_ok=True)

# ---- Sentiment by Time ----
t = pd.read_csv(f"{PBI}/sentiment_by_time.csv")
t.to_json(f"{OUT}/sentiment_by_time.json", orient="records")

# ---- Sentiment by Age ----
a = pd.read_csv(f"{PBI}/sentiment_by_age.csv")
a.to_json(f"{OUT}/sentiment_by_age.json", orient="records")

# ---- Sentiment by Country (top 15 by volume) ----
c = pd.read_csv(f"{PBI}/sentiment_by_country.csv")
totals = c.groupby("country")["count"].sum().sort_values(ascending=False).head(15).index
c_top = c[c["country"].isin(totals)]
c_top.to_json(f"{OUT}/sentiment_by_country.json", orient="records")

# ---- Top Words by Sentiment ----
w = pd.read_csv(f"{PBI}/top_words_by_sentiment.csv")
w_top = w.groupby("sentiment").head(15)
w_top.to_json(f"{OUT}/top_words.json", orient="records")

# ---- Model Performance ----
m = pd.read_csv(f"{PBI}/model_performance.csv")
m.to_json(f"{OUT}/model_performance.json", orient="records")

# ---- Predictions: accuracy donut ----
p = pd.read_csv(f"{PBI}/predictions.csv")
donut = p["is_correct"].value_counts().reset_index()
donut.columns = ["is_correct", "count"]
donut.to_json(f"{OUT}/accuracy_donut.json", orient="records")

# ---- Predictions: actual vs predicted ----
avp = p.groupby(["actual_sentiment", "predicted_sentiment"]).size().reset_index(name="count")
avp.to_json(f"{OUT}/actual_vs_predicted.json", orient="records")

# ---- Predictions: avg confidence by predicted sentiment ----
conf = p.groupby("predicted_sentiment")["confidence"].mean().reset_index()
conf.columns = ["predicted_sentiment", "avg_confidence"]
conf.to_json(f"{OUT}/confidence_by_sentiment.json", orient="records")

print("Regenerated chart data in static/data/")

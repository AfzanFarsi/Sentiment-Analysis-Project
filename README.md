# Project 11 — Sentiment Analysis Using Machine Learning

## 1. Project Goal
**Type:** Multi-class text **classification** (NLP).
**Business problem:** Businesses receive large volumes of unstructured text — tweets, reviews,
survey responses — and need a fast, scalable way to know whether feedback is **positive,
negative, or neutral**. Manually reading everything doesn't scale. This project builds a
classifier that reads raw text and predicts sentiment automatically, then feeds the results
into a Power BI dashboard so non-technical stakeholders can track sentiment trends by time
of day, age group, and country without touching code.

## 2. Data
Source: pre-supplied `train.csv` (27,480 raw rows) / `test.csv` (4,815 raw rows) — a Twitter
Sentiment Extraction style dataset enriched with demographic fields.

| Column | Description |
|---|---|
| `textID` | Unique tweet identifier |
| `text` | Raw tweet text |
| `selected_text` | Span of text driving the sentiment (train only) |
| `sentiment` | Target label: negative / neutral / positive |
| `Time of Tweet` | morning / noon / night |
| `Age of User` | Age bracket of the poster |
| `Country` | Country of the poster |
| `Population -2020`, `Land Area (Km²)`, `Density (P/Km²)` | Country-level demographic stats |

## 3. Quickest way to run it: one script
`run_pipeline.py` (in the project root) runs all five steps below in sequence in a
single command. From the project root:
```bash
pip install -r requirements.txt
python run_pipeline.py
```
It uses paths relative to its own location, so it works no matter which folder you
run it from. Takes about 1–2 minutes on a laptop. If you'd rather run each stage
separately (e.g. to inspect intermediate output), use the individual scripts below instead.

## 3b. Pipeline (individual scripts, run in order)

| Script | Step | What it does |
|---|---|---|
| `scripts/01_preprocessing.py` | Preprocess Data | Drops rows with missing text/label (1,281 fully-blank rows in test.csv), removes duplicates, lowercases + strips URLs/mentions/punctuation, removes stopwords, encodes sentiment (0/1/2), cleans demographic columns, caps outliers in population/density at the 1st/99th percentile |
| `scripts/02_eda.py` | EDA | 8 charts: class balance, word-count distributions, top words per class, sentiment by time-of-day/age/country, numeric correlation heatmap, text-length outlier boxplot |
| `scripts/03_feature_engineering.py` | Engineer Features | TF-IDF (5,000 features, uni+bi-grams), TruncatedSVD (100 components, ~21% variance retained, saved for lightweight downstream use), one-hot encoding of categoricals, frequency-encoding of `Country`, Min-Max scaled numeric features, all combined into one sparse matrix |
| `scripts/04_model_building.py` | Build ML Model | Trains Logistic Regression, Random Forest, and Multinomial Naive Bayes on an internal train/validation split, selects the best by macro-F1, refits on full training data, evaluates on the held-out `test.csv` |
| `scripts/05_powerbi_export.py` | Visualize in Power BI | Flattens predictions + aggregates into Power-BI-ready CSVs |

Run them in order from inside `scripts/`:
```bash
python3 01_preprocessing.py
python3 02_eda.py
python3 03_feature_engineering.py
python3 04_model_building.py
python3 05_powerbi_export.py
```

## 4. Results

**Best model: Logistic Regression** (selected by macro-F1 on validation split)

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Validation (15% of train) | 68.3% | — | — | 68.5% |
| **Test (held-out test.csv)** | **70.0%** | 70.2% | 70.2% | 70.3% |

Per-class performance on the test set:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.67 | 0.71 | 0.69 |
| Neutral | 0.67 | 0.65 | 0.66 |
| Positive | 0.76 | 0.75 | 0.75 |

Random Forest and Naive Bayes both underperformed Logistic Regression on this sparse
high-dimensional TF-IDF feature space (val accuracy 62.2% and 63.9% respectively) —
expected, since linear models tend to handle sparse bag-of-words features better than
tree ensembles, and the "positive" class is the easiest to separate (praise words like
"good", "love", "great" are strong, unambiguous signals) while "neutral" is hardest to
call because it overlaps lexically with both other classes.

Note: `Time of Tweet`, `Age of User`, `Country`, and the population/density fields turned
out to carry almost no linear correlation with sentiment (|r| < 0.03) — sentiment here is
driven almost entirely by word choice, not demographics. They're kept in the model and the
dashboard anyway since they're useful **breakdown dimensions** even without predictive power.

## 5. Folder Contents

```
Sentiment_Analysis_Project/
├── README.md                      <- this file
├── data/
│   ├── raw/                       <- original train.csv / test.csv
│   └── processed/                 <- cleaned CSVs + saved feature matrices (.npz/.npy)
├── scripts/                       <- 01–05, run in order (see above)
├── models/                        <- fitted vectorizer, encoders, best model (.joblib), metrics_summary.json
├── visuals/                       <- 11 PNG charts from EDA + model evaluation
├── powerbi_exports/                <- flat CSVs ready to load into Power BI
│   ├── predictions.csv             <- row-level: text, actual vs predicted, confidence, demographics
│   ├── sentiment_by_time.csv       <- sentiment mix by time of day (trend visual)
│   ├── sentiment_by_age.csv        <- sentiment mix by age group
│   ├── sentiment_by_country.csv    <- sentiment mix by top-30 countries (map visual)
│   ├── top_words_by_sentiment.csv  <- word-impact table (word, sentiment, TF-IDF weight)
│   └── model_performance.csv       <- accuracy/precision/recall/F1 per model (KPI cards)
└── reports/                        <- (optional) exported Power BI screenshots / write-up
```

## 6. Building the Power BI Dashboard
Load the 6 CSVs in `powerbi_exports/` into Power BI Desktop (Get Data → Text/CSV) and build:
1. **KPI cards** from `model_performance.csv` — accuracy / F1 of the deployed model.
2. **Stacked bar / trend** from `sentiment_by_time.csv` — sentiment flow across the day.
3. **Stacked bar** from `sentiment_by_age.csv` — sentiment by age group.
4. **Filled map** from `sentiment_by_country.csv` — sentiment mix by country (use `country` as
   the map field).
5. **Word-impact bar chart / packed bubble** from `top_words_by_sentiment.csv`, sliced by
   `sentiment` — shows which words most drive each sentiment class.
6. **Table/matrix** from `predictions.csv` with a slicer on `is_correct` and `predicted_sentiment`
   to let users drill into individual misclassified tweets.
Relate the tables on shared dimensions (`time_of_tweet`, `age_group`, `country`, `sentiment`)
to enable cross-filtering across visuals.

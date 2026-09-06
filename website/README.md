# Sentiscope — Sentiment Analysis Website

A four-page Flask website:

1. **Home** (`/`) — welcome page with a short pitch and links into the other three pages.
2. **Dashboard** (`/dashboard`) — live charts (Chart.js), rebuilt from the same fields
   as your Power BI report's 6 pages (Sentiment_by_age, Sentiment_by_time,
   Sentiment_by_country, Top_Words_by_Sentiment, predictions, Model_Performance),
   with a tab bar to switch between them. Data comes straight from
   `powerbi_exports/*.csv`, not screenshots.
3. **Predict** (`/predict`) — the live demo. Type a sentence, click "Take a reading,"
   and a Logistic Regression model classifies it as negative / neutral / positive in
   real time, with an animated gauge and the specific words that drove the call.
4. **About** (`/about`) — the five-step pipeline behind the model, plus notes on the
   data, dashboard, and tech stack.

Every subpage has a "Back to home" link (in the header nav and in the footer).

## Folder contents
```
website/
├── app.py                       <- Flask backend (routes + live prediction API)
├── requirements.txt
├── model/
│   ├── train_text_model.py      <- trains the text-only model used for live demo
│   ├── text_tfidf.joblib        <- (generated) TF-IDF vectorizer
│   ├── text_sentiment_model.joblib  <- (generated) trained classifier
│   └── text_model_metrics.json  <- (generated) accuracy/F1 shown on the home page
├── templates/
│   ├── base.html                <- shared header/nav/footer
│   ├── home.html
│   ├── dashboard.html            <- tab bar + 6 chart pages
│   ├── predict.html
│   └── about.html
└── static/
    ├── css/style.css
    ├── js/main.js                <- live-demo gauge logic (Predict page)
    ├── js/dashboard.js           <- builds all 6 dashboard chart pages
    └── data/*.json               <- (generated) chart data exported from powerbi_exports/
```

## Setup (from the `website/` folder)

```bash
pip install -r requirements.txt
```

**If `model/*.joblib` files aren't already there**, generate them once:
```bash
cd model
python train_text_model.py
cd ..
```
This reads `../data/processed/train_clean.csv` and `test_clean.csv`, so run the
main pipeline (`../run_pipeline.py`) first if those don't exist yet.

**Chart data** in `static/data/*.json` is already generated from `../powerbi_exports/`.
If you re-run the main pipeline and want the dashboard to reflect new numbers, regenerate
it with `regenerate_chart_data.py` in this folder.

## Run it

```bash
python app.py
```
Then open **http://127.0.0.1:5000**.

Note: the page pulls Google Fonts and Chart.js from a CDN, so it needs an internet
connection in the browser (the Flask server itself runs fully offline/locally).

## How the live demo differs from the main model
The main project model (`../models/best_sentiment_model.joblib`) uses tweet text
*plus* demographic fields (time of day, age group, country, population stats).
A website visitor typing into a text box doesn't have any of that, so this site
trains and serves a second, text-only model (`model/train_text_model.py`) —
same algorithm (Logistic Regression + TF-IDF), same training data, just without
the demographic features.

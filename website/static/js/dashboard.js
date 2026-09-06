// =========================================================
// SENTISCOPE — dashboard.js
// Builds live Chart.js visuals from the exported CSV -> JSON
// data in static/data/, mirroring the 6 pages of the Power BI
// report (Sentiment_by_age, Sentiment_by_time, Sentiment_by_country,
// Top_Words_by_Sentiment, predictions, Model_Performance).
// =========================================================

const SENTIMENT_COLORS = {
  negative: "#C1443B",
  neutral: "#6B7280",
  positive: "#1F9D74",
};

const ORDER = ["negative", "neutral", "positive"];

const baseScales = {
  x: { ticks: { font: { family: "IBM Plex Mono", size: 10 } }, grid: { display: false } },
  y: { ticks: { font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#E4E6EA" } },
};

async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

/* ---------------- tab switching ---------------- */
document.querySelectorAll(".dash-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".dash-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const page = tab.dataset.page;
    document.querySelectorAll(".dash-page").forEach((p) => {
      p.hidden = p.dataset.page !== page;
    });
  });
});

/* ---------------- Sentiment by Age ---------------- */
async function buildAgeChart() {
  const raw = await loadJSON("/static/data/sentiment_by_age.json");
  const groups = ["0-20", "21-30", "31-45", "46-60", "60-70", "70-100"];
  const datasets = ORDER.map((s) => ({
    label: s,
    data: groups.map((g) => {
      const row = raw.find((d) => d.age_group === g && d.predicted_sentiment === s);
      return row ? row.count : 0;
    }),
    backgroundColor: SENTIMENT_COLORS[s],
  }));
  new Chart(document.getElementById("chart-age"), {
    type: "bar",
    data: { labels: groups, datasets },
    options: {
      plugins: { legend: { position: "bottom", labels: { font: { family: "Inter", size: 11 } } } },
      scales: { x: { ...baseScales.x, stacked: true }, y: { ...baseScales.y, stacked: true } },
    },
  });
}

/* ---------------- Sentiment by Time ---------------- */
async function buildTimeChart() {
  const raw = await loadJSON("/static/data/sentiment_by_time.json");
  const groups = ["morning", "noon", "night"];
  const datasets = ORDER.map((s) => ({
    label: s,
    data: groups.map((g) => {
      const row = raw.find((d) => d.time_of_tweet === g && d.predicted_sentiment === s);
      return row ? row.count : 0;
    }),
    backgroundColor: SENTIMENT_COLORS[s],
  }));
  new Chart(document.getElementById("chart-time"), {
    type: "bar",
    data: { labels: groups, datasets },
    options: {
      plugins: { legend: { position: "bottom", labels: { font: { family: "Inter", size: 11 } } } },
      scales: { x: { ...baseScales.x, stacked: true }, y: { ...baseScales.y, stacked: true } },
    },
  });
}

/* ---------------- Sentiment by Country ---------------- */
async function buildCountryChart() {
  const raw = await loadJSON("/static/data/sentiment_by_country.json");
  const totals = {};
  raw.forEach((d) => { totals[d.country] = (totals[d.country] || 0) + d.count; });
  const countries = Object.entries(totals).sort((a, b) => b[1] - a[1]).map((d) => d[0]);

  const datasets = ORDER.map((s) => ({
    label: s,
    data: countries.map((c) => {
      const row = raw.find((d) => d.country === c && d.predicted_sentiment === s);
      return row ? row.count : 0;
    }),
    backgroundColor: SENTIMENT_COLORS[s],
  }));

  new Chart(document.getElementById("chart-country"), {
    type: "bar",
    data: { labels: countries, datasets },
    options: {
      indexAxis: "y",
      plugins: { legend: { position: "bottom", labels: { font: { family: "Inter", size: 11 } } } },
      scales: {
        x: { ...baseScales.x, stacked: true, grid: { color: "#E4E6EA" } },
        y: { ...baseScales.y, stacked: true, ticks: { font: { family: "Inter", size: 11 } }, grid: { display: false } },
      },
    },
  });
}

/* ---------------- Top Words ---------------- */
let wordsChart = null;
async function buildWordsChart(sentiment) {
  const raw = await loadJSON("/static/data/top_words.json");
  const filtered = raw.filter((d) => d.sentiment === sentiment).sort((a, b) => b.weight - a.weight).slice(0, 15);

  if (wordsChart) wordsChart.destroy();
  wordsChart = new Chart(document.getElementById("chart-words"), {
    type: "bar",
    data: {
      labels: filtered.map((d) => d.word),
      datasets: [{ data: filtered.map((d) => d.weight), backgroundColor: SENTIMENT_COLORS[sentiment] }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: baseScales.x,
        y: { ticks: { font: { family: "IBM Plex Mono", size: 11 } }, grid: { display: false } },
      },
    },
  });
}

document.querySelectorAll(".word-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".word-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    buildWordsChart(tab.dataset.sentiment);
  });
});

/* ---------------- Predictions: accuracy donut ---------------- */
async function buildAccuracyDonut() {
  const raw = await loadJSON("/static/data/accuracy_donut.json");
  const correct = raw.find((d) => d.is_correct === true);
  const incorrect = raw.find((d) => d.is_correct === false);
  new Chart(document.getElementById("chart-accuracy-donut"), {
    type: "doughnut",
    data: {
      labels: ["Correct", "Incorrect"],
      datasets: [{
        data: [correct ? correct.count : 0, incorrect ? incorrect.count : 0],
        backgroundColor: ["#2451E0", "#D8DCE3"],
        borderWidth: 0,
      }],
    },
    options: {
      plugins: { legend: { position: "bottom", labels: { font: { family: "Inter", size: 11 } } } },
      cutout: "62%",
    },
  });
}

/* ---------------- Predictions: actual vs predicted ---------------- */
async function buildActualPredictedChart() {
  const raw = await loadJSON("/static/data/actual_vs_predicted.json");
  const datasets = ORDER.map((s) => ({
    label: s,
    data: ORDER.map((actual) => {
      const row = raw.find((d) => d.actual_sentiment === actual && d.predicted_sentiment === s);
      return row ? row.count : 0;
    }),
    backgroundColor: SENTIMENT_COLORS[s],
  }));
  new Chart(document.getElementById("chart-actual-predicted"), {
    type: "bar",
    data: { labels: ORDER, datasets },
    options: {
      plugins: { legend: { position: "bottom", labels: { font: { family: "Inter", size: 11 } } } },
      scales: baseScales,
    },
  });
}

/* ---------------- Predictions: confidence by sentiment ---------------- */
async function buildConfidenceChart() {
  const raw = await loadJSON("/static/data/confidence_by_sentiment.json");
  const labels = ORDER;
  const data = labels.map((s) => {
    const row = raw.find((d) => d.predicted_sentiment === s);
    return row ? Math.round(row.avg_confidence * 100) / 100 : 0;
  });
  new Chart(document.getElementById("chart-confidence"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ data, backgroundColor: labels.map((s) => SENTIMENT_COLORS[s]) }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: baseScales.x, y: { ...baseScales.y, max: 1 } },
    },
  });
}

/* ---------------- Model Performance ---------------- */
async function buildModelCharts() {
  const raw = await loadJSON("/static/data/model_performance.json");
  const val = raw.filter((d) => d.dataset === "validation");

  new Chart(document.getElementById("chart-model-accuracy"), {
    type: "bar",
    data: {
      labels: val.map((d) => d.model),
      datasets: [{ data: val.map((d) => Math.round(d.accuracy * 100)), backgroundColor: "#2451E0" }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { font: { family: "Inter", size: 10 } }, grid: { display: false } }, y: { ...baseScales.y, max: 100, ticks: { font: { family: "IBM Plex Mono", size: 10 }, callback: (v) => v + "%" } } },
    },
  });

  new Chart(document.getElementById("chart-model-f1"), {
    type: "bar",
    data: {
      labels: val.map((d) => d.model),
      datasets: [{ data: val.map((d) => Math.round(d.f1_macro * 100)), backgroundColor: "#8FA6F0" }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { font: { family: "Inter", size: 10 } }, grid: { display: false } }, y: { ...baseScales.y, max: 100, ticks: { font: { family: "IBM Plex Mono", size: 10 }, callback: (v) => v + "%" } } },
    },
  });
}

/* ---------------- init ---------------- */
buildAgeChart();
buildTimeChart();
buildCountryChart();
buildWordsChart("negative");
buildAccuracyDonut();
buildActualPredictedChart();
buildConfidenceChart();
buildModelCharts();

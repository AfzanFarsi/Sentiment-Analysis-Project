// =========================================================
// SENTISCOPE — main.js
// Handles the live demo on the Predict page: gauge needle,
// probability bars, and word-contribution chips.
// (The Dashboard page uses static Power BI screenshots, so no
// charting library is needed here.)
// =========================================================

const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const errorMsg = document.getElementById("error-msg");
const needle = document.getElementById("needle");
const readoutSentiment = document.getElementById("readout-sentiment");
const readoutConfidence = document.getElementById("readout-confidence");
const wordContributions = document.getElementById("word-contributions");
const wordChips = document.getElementById("word-chips");

// Only run the demo logic on pages that actually have these elements (i.e. /predict)
if (textInput && analyzeBtn) {

  textInput.addEventListener("input", () => {
    charCount.textContent = textInput.value.length;
  });

  const setNeedle = (negProb, neuProb, posProb) => {
    // Map probability-weighted position to an angle from -85 to +85 degrees.
    // score in [-1, 1]: -1 = fully negative, 0 = fully neutral, +1 = fully positive
    const score = posProb - negProb;
    const angle = score * 85;
    needle.style.transform = `rotate(${angle}deg)`;
  };

  const setProbaBars = (probs) => {
    document.querySelectorAll(".proba-row").forEach((row) => {
      const key = row.dataset.key;
      const pct = Math.round((probs[key] || 0) * 100);
      row.querySelector(".proba-fill").style.width = pct + "%";
      row.querySelector(".proba-pct").textContent = pct + "%";
    });
  };

  const analyzeText = async () => {
    const text = textInput.value.trim();
    errorMsg.hidden = true;

    if (!text) {
      errorMsg.textContent = "Type something first — even a sentence will do.";
      errorMsg.hidden = false;
      return;
    }

    analyzeBtn.disabled = true;
    const originalLabel = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = "<span>Reading&hellip;</span>";

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();

      if (!res.ok) {
        errorMsg.textContent = data.error || "Something went wrong. Try again.";
        errorMsg.hidden = false;
        return;
      }

      readoutSentiment.textContent = data.sentiment;
      readoutSentiment.className = "readout-sentiment is-" + data.sentiment;
      readoutConfidence.textContent = Math.round(data.confidence * 100) + "% confident";

      setNeedle(data.probabilities.negative, data.probabilities.neutral, data.probabilities.positive);
      setProbaBars(data.probabilities);

      if (data.top_words && data.top_words.length > 0) {
        wordChips.innerHTML = "";
        data.top_words.forEach((w) => {
          const chip = document.createElement("span");
          chip.className = "word-chip is-" + data.sentiment;
          chip.textContent = w.word;
          wordChips.appendChild(chip);
        });
        wordContributions.hidden = false;
      } else {
        wordContributions.hidden = true;
      }
    } catch (err) {
      errorMsg.textContent = "Couldn't reach the server. Is app.py running?";
      errorMsg.hidden = false;
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = originalLabel;
    }
  };

  analyzeBtn.addEventListener("click", analyzeText);
  textInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyzeText();
  });
}

"""
Mental Health Sentiment Analyzer — Flask Web App
=================================================
Run:
    pip install flask deep-translator indic-transliteration
    python app.py

Then open: http://127.0.0.1:5000
"""

import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, render_template_string, request
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from scipy.sparse import hstack

# ─── Load model ──────────────────────────────────────────────
MODEL_DIR = "mental_health_model"
HISTORY_FILE = "history.json"
CLASS_LABELS = ["positive", "negative", "neutral"]


def load_artifacts():
    arts = {}
    for name in ("model", "tfidf_word", "tfidf_char", "metadata"):
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "rb") as f:
            arts[name] = pickle.load(f)
    return arts


ARTIFACTS = load_artifacts()
print(f"✅ Model loaded — accuracy: {ARTIFACTS['metadata']['accuracy'] * 100:.2f}%")

# ─── Inference ───────────────────────────────────────────────
NEGATIVE_KEYWORDS = [
    "hopeless", "tired", "no energy", "can't eat", "haven't eaten",
    "crying", "meaningless", "don't care", "depressed", "worthless",
    "lonely", "anxious", "overwhelmed", "no motivation", "give up",
    "exhausted", "emptiness", "empty", "doesn't matter", "does not matter",
    "nothing matters", "no point", "numb", "flat", "disconnected",
]

SEVERE_WORDS = [
    "hopeless", "worthless", "meaningless", "give up", "no point",
    "nothing matters", "depressed", "can't go on", "end it all",
]

NEUTRAL_PATTERNS = [
    "nothing particularly", "routine", "nothing stood out", "don't feel strongly",
    "no strong feelings", "just normal", "average day", "went back to sleep",
    "same as usual", "ordinary day", "pretty normal", "nothing much happened",
]

EMOTION_KEYWORDS = {
    "fatigue": {
        "weight": 2,
        "keywords": [
            "tired", "exhausted", "burnt out", "burned out", "drained",
            "worn out", "fatigue", "no energy", "low energy", "sleepy",
        ],
    },
    "sadness": {
        "weight": 2,
        "keywords": [
            "sad", "hopeless", "crying", "empty", "lonely", "down",
            "meaningless", "worthless", "numb", "low",
        ],
    },
    "anxiety": {
        "weight": 2,
        "keywords": [
            "anxious", "nervous", "worried", "overthinking", "panic",
            "stressed", "stress", "overwhelmed", "fear", "uneasy",
        ],
    },
    "joy": {
        "weight": 2,
        "keywords": [
            "happy", "excited", "great", "grateful", "joy", "relieved",
            "good", "calm", "proud", "hopeful",
        ],
    },
    "neutral": {
        "weight": 1,
        "keywords": [
            "routine", "nothing", "normal", "usual", "average", "fine",
            "okay", "ok", "steady", "same",
        ],
    },
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\[NAME\]", "someone", text)
    text = re.sub(r"[^a-z\s'!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transliterate_text(text, lang):
    try:
        if lang in ["hi", "mr"]:
            return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return text
    except Exception:
        return text


def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def normalize_scores(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = np.clip(scores, -8.0, 8.0)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def override_negative(text):
    text = text.lower()
    return any(word in text for word in NEGATIVE_KEYWORDS)


def contrast_negative(text):
    text = text.lower()
    if "but" in text:
        after_but = text.split("but", 1)[1]
        return any(word in after_but for word in NEGATIVE_KEYWORDS)
    return False


def severe_negative(text):
    text = text.lower()
    return any(word in text for word in SEVERE_WORDS)


def force_neutral(text):
    text = text.lower()
    return any(pattern in text for pattern in NEUTRAL_PATTERNS)


def detect_emotion(text):
    text = text.lower()
    scores = {name: 0 for name in EMOTION_KEYWORDS}

    for emotion, config in EMOTION_KEYWORDS.items():
        for keyword in config["keywords"]:
            if keyword in text:
                scores[emotion] += config["weight"]

    top_score = max(scores.values())
    if top_score == 0:
        return "mixed"

    top_emotions = [emotion for emotion, score in scores.items() if score == top_score]
    if len(top_emotions) > 1 and "neutral" not in top_emotions:
        return "mixed"
    return top_emotions[0]


def get_intensity(confidence):
    if confidence >= 85:
        return "strong"
    if confidence >= 65:
        return "moderate"
    return "slight"


def generate_interpretation(sentiment_class, intensity, emotion):
    if sentiment_class == "negative":
        if emotion == "fatigue":
            return f"You seem mentally tired or drained, with a {intensity} negative tone. Rest and smaller tasks may help."
        if emotion == "sadness":
            return f"Your text carries a {intensity} low mood, suggesting emotional heaviness or discouragement."
        if emotion == "anxiety":
            return f"There are signs of a {intensity} anxious state, with stress or worry showing through."
        if emotion == "mixed":
            return f"Your message suggests a {intensity} difficult emotional state with overlapping negative signals."
        return f"The text shows a {intensity} negative tone that may reflect emotional strain."

    if sentiment_class == "positive":
        if emotion == "joy":
            return f"Your text reflects a {intensity} positive state with signs of joy, relief, or encouragement."
        return f"The overall tone feels {intensity}ly positive and emotionally supportive."

    if emotion == "neutral":
        return "Your day appears stable and emotionally balanced, though it may feel routine or low in variation."
    return "The text appears emotionally steady overall, without strong positive or negative signals."


def generate_suggestion(sentiment_class, emotion):
    if sentiment_class == "negative":
        if emotion == "fatigue":
            return "Try rest, hydration, and reducing one demand at a time."
        if emotion == "anxiety":
            return "Try slow breathing, grounding, or focusing on one manageable next step."
        if emotion == "sadness":
            return "Journaling or reaching out to someone you trust may help lighten the load."
        return "Try taking breaks, journaling, or speaking with someone you trust."

    if sentiment_class == "positive":
        return "Keep engaging in the people, habits, or activities that are supporting your well-being."

    return "Consider adding a small meaningful activity, a short walk, or a change in routine for variety."


def format_sentiment(sentiment_class, intensity):
    if sentiment_class == "neutral":
        return "neutral"
    return f"{intensity} {sentiment_class}"


def titleize_sentiment(sentiment: str) -> str:
    return " ".join(word.capitalize() for word in sentiment.split())


def predict_model_sentiment(cleaned: str):
    X = hstack([
        ARTIFACTS["tfidf_word"].transform([cleaned]),
        ARTIFACTS["tfidf_char"].transform([cleaned]),
    ])

    raw_sentiment = ARTIFACTS["model"].predict(X)[0]
    raw_scores = ARTIFACTS["model"].decision_function(X)
    probs = normalize_scores(raw_scores)[0]
    classes = list(getattr(ARTIFACTS["model"], "classes_", ARTIFACTS["metadata"].get("classes", CLASS_LABELS)))

    if raw_sentiment in classes:
        idx = classes.index(raw_sentiment)
    else:
        idx = int(np.argmax(probs))
        raw_sentiment = classes[idx]

    confidence = float(probs[idx] * 100)
    return raw_sentiment, confidence


def apply_rule_overrides(text: str, sentiment_class: str, confidence: float, emotion: str):
    if force_neutral(text):
        return "neutral", min(confidence, 60.0)

    if override_negative(text) or contrast_negative(text):
        sentiment_class = "negative"
        confidence = max(confidence, 67.0)

    if severe_negative(text):
        sentiment_class = "negative"
        confidence = max(confidence, 88.0)

    if emotion in {"fatigue", "sadness", "anxiety"} and sentiment_class == "positive":
        sentiment_class = "negative"
        confidence = min(max(confidence, 58.0), 72.0)

    if emotion == "neutral" and confidence < 70:
        sentiment_class = "neutral"
        confidence = min(confidence, 64.0)

    if emotion == "mixed" and confidence < 58:
        sentiment_class = "neutral"

    return sentiment_class, confidence


def predict_sentiment(text: str, language: str = "en") -> dict:
    original_text = text
    devanagari_text = ""
    translated_text = text

    if language in {"hi", "mr"}:
        devanagari_text = transliterate_text(text, language)
        translated_text = translate_to_english(devanagari_text)
    else:
        translated_text = text

    cleaned = clean_text(translated_text)

    if not cleaned:
        return {
            "original_text": original_text,
            "devanagari_text": devanagari_text,
            "translated_text": translated_text,
            "language": language,
            "sentiment": "unknown",
            "sentiment_class": "neutral",
            "confidence": 0.0,
            "emotion": "neutral",
            "intensity": "slight",
            "interpretation": "No meaningful text was provided.",
            "suggestion": "Try entering a sentence or short paragraph for analysis.",
            "label": "Unknown",
        }

    sentiment_class, confidence = predict_model_sentiment(cleaned)
    emotion = detect_emotion(translated_text)
    sentiment_class, confidence = apply_rule_overrides(translated_text, sentiment_class, confidence, emotion)

    intensity = get_intensity(confidence)
    sentiment = format_sentiment(sentiment_class, intensity)
    interpretation = generate_interpretation(sentiment_class, intensity, emotion)
    suggestion = generate_suggestion(sentiment_class, emotion)

    return {
        "original_text": original_text,
        "devanagari_text": devanagari_text,
        "translated_text": translated_text,
        "language": language,
        "sentiment": sentiment,
        "sentiment_class": sentiment_class,
        "confidence": round(confidence, 1),
        "emotion": emotion,
        "intensity": intensity,
        "interpretation": interpretation,
        "suggestion": suggestion,
        "label": titleize_sentiment(sentiment),
    }


# ─── Persistent History ──────────────────────────────────────
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_history(entry):
    history = load_history()
    history.append(entry)
    history = history[-30:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ─── Flask app ───────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MindRead — Mental Health Sentiment Analyzer</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --border:    #1e2330;
    --text:      #e8eaf0;
    --muted:     #5a6075;
    --pos:       #4ade80;
    --pos-dim:   #14532d;
    --neg:       #f87171;
    --neg-dim:   #7f1d1d;
    --neu:       #fbbf24;
    --neu-dim:   #78350f;
    --accent:    #818cf8;
    --radius:    16px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 20px 80px;
    overflow-x: hidden;
  }

  body::before, body::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    pointer-events: none;
    z-index: 0;
  }
  body::before {
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(79,70,229,.18), transparent 70%);
    top: -200px; left: -150px;
  }
  body::after {
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(74,222,128,.10), transparent 70%);
    bottom: -200px; right: -100px;
  }

  .wrap { width: 100%; max-width: 680px; position: relative; z-index: 1; }

  header { text-align: center; margin-bottom: 52px; }
  .logo {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 20px;
  }
  .logo-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
  h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 6vw, 3.2rem);
    font-weight: 700; line-height: 1.1;
    letter-spacing: -0.02em;
  }
  h1 em { font-style: italic; color: var(--accent); }
  .subtitle {
    margin-top: 14px; color: var(--muted);
    font-size: 15px; line-height: 1.6;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    margin-bottom: 20px;
  }

  label {
    display: block;
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 10px;
  }

  select,
  textarea {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    outline: none;
    transition: border-color .2s;
  }

  select {
    padding: 14px 16px;
    margin-bottom: 14px;
  }

  textarea {
    min-height: 140px;
    line-height: 1.7;
    padding: 16px 18px;
    resize: vertical;
  }

  textarea::placeholder { color: var(--muted); }
  textarea:focus,
  select:focus { border-color: var(--accent); }

  .input-actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 10px;
    gap: 10px;
  }

  .mic-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 8px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: border-color .2s, opacity .2s;
  }
  .mic-btn:hover { border-color: var(--accent); }
  .mic-btn.active {
    border-color: var(--accent);
    color: var(--accent);
  }

  .char-count {
    text-align: right; font-size: 12px; color: var(--muted);
    font-family: 'DM Mono', monospace; margin-top: 8px;
  }

  .btn {
    width: 100%; margin-top: 20px;
    padding: 16px;
    background: var(--accent);
    color: #fff;
    border: none; border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px; font-weight: 500;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
    position: relative; overflow: hidden;
  }
  .btn:hover { opacity: .88; }
  .btn:active { transform: scale(.98); }
  .btn.loading { pointer-events: none; opacity: .7; }
  .btn .spinner {
    display: none; width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin .7s linear infinite;
    margin: 0 auto;
  }
  .btn.loading .btn-text { display: none; }
  .btn.loading .spinner { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .result {
    display: none;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    overflow: hidden;
    animation: slideUp .35s ease;
  }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result.visible { display: block; }

  .result-header {
    padding: 24px 32px;
    display: flex; align-items: center; gap: 20px;
  }
  .result-header.positive { background: var(--pos-dim); border-bottom: 1px solid #166534; }
  .result-header.negative { background: var(--neg-dim); border-bottom: 1px solid #991b1b; }
  .result-header.neutral { background: var(--neu-dim); border-bottom: 1px solid #a16207; }

  .result-icon { font-size: 2.4rem; line-height: 1; }

  .result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem; font-weight: 700;
  }
  .result-header.positive .result-label { color: var(--pos); }
  .result-header.negative .result-label { color: var(--neg); }
  .result-header.neutral .result-label { color: var(--neu); }

  .result-tag {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 11px; letter-spacing: .1em;
    padding: 4px 10px; border-radius: 20px;
  }
  .result-header.positive .result-tag { background: #166534; color: var(--pos); }
  .result-header.negative .result-tag { background: #991b1b; color: var(--neg); }
  .result-header.neutral .result-tag { background: #a16207; color: var(--neu); }

  .result-body { background: var(--surface); padding: 24px 32px; }

  .conf-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .conf-label { font-size: 13px; color: var(--muted); font-family: 'DM Mono', monospace; letter-spacing: .08em; }
  .conf-value { font-size: 13px; font-family: 'DM Mono', monospace; }

  .bar-track {
    height: 6px; background: var(--border); border-radius: 99px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 99px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
  }
  .positive .bar-fill { background: var(--pos); }
  .negative .bar-fill { background: var(--neg); }
  .neutral .bar-fill { background: var(--neu); }

  .interpretation {
    margin-top: 20px; padding-top: 20px;
    border-top: 1px solid var(--border);
    font-size: 14px; line-height: 1.7; color: var(--muted);
  }
  .interpretation strong { color: var(--text); font-weight: 500; }

  .history-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: .15em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 12px;
  }
  .history-list { display: flex; flex-direction: column; gap: 8px; }
  .history-item {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px;
    display: flex; align-items: center; gap: 12px;
    cursor: pointer; transition: border-color .2s;
  }
  .history-item:hover { border-color: var(--accent); }
  .history-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }
  .history-dot.positive { background: var(--pos); }
  .history-dot.negative { background: var(--neg); }
  .history-dot.neutral { background: var(--neu); }
  .history-text { flex: 1; font-size: 13px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .history-conf { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }

  footer {
    margin-top: 48px; text-align: center;
    font-size: 12px; color: var(--muted);
    font-family: 'DM Mono', monospace; letter-spacing: .08em;
  }
  footer span { color: var(--accent); }

  .error-msg {
    display: none; background: #2d1515;
    border: 1px solid #7f1d1d; border-radius: 10px;
    padding: 14px 18px; margin-top: 12px;
    font-size: 14px; color: var(--neg);
  }
  .error-msg.visible { display: block; }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo"><div class="logo-dot"></div>Mental Health AI<div class="logo-dot"></div></div>
    <h1>Understand how you<br/><em>truly</em> feel</h1>
    <p class="subtitle">Paste any text — a journal entry, message, or thought —<br/>and the model will detect its emotional sentiment.</p>
  </header>

  <div class="card">
    <label for="languageSelect">Language</label>
    <select id="languageSelect">
      <option value="en">English</option>
      <option value="hi">Hindi</option>
      <option value="mr">Marathi</option>
    </select>

    <label for="txt">Your text</label>
    <textarea id="txt" placeholder="e.g. I've been feeling really overwhelmed lately and I don't know how to cope..."></textarea>
    <div class="input-actions">
      <button class="mic-btn" id="translateBtn" type="button">🌐 Translate</button>
      <button class="mic-btn" id="micBtn" type="button" onclick="startSpeechInput()">
        <span>🎙</span>
        <span>Speak</span>
      </button>
    </div>
    <div class="char-count"><span id="charCount">0</span> characters</div>
    <div class="error-msg" id="errMsg"></div>
    <button class="btn" id="analyzeBtn" onclick="analyze()">
      <span class="btn-text">Analyze Sentiment</span>
      <div class="spinner"></div>
    </button>
  </div>

  <div class="result" id="result">
    <div class="result-header" id="resultHeader">
      <div class="result-icon" id="resultIcon"></div>
      <div class="result-label" id="resultLabel"></div>
      <div class="result-tag" id="resultTag"></div>
    </div>
    <div class="result-body">
      <div class="conf-row">
        <span class="conf-label">Confidence</span>
        <span class="conf-value" id="confValue"></span>
      </div>
      <div class="bar-track" id="barTrack">
        <div class="bar-fill" id="barFill" style="width:0%"></div>
      </div>
      <div class="interpretation" id="interpretation"></div>
    </div>
  </div>

  <div id="historySection" style="display:none; margin-top:28px;">
    <div class="history-title">Recent analyses</div>
    <div class="history-list" id="historyList"></div>
  </div>

  <footer>
    Model accuracy <span>63.61%</span> &nbsp;·&nbsp; LinearSVC + TF-IDF &nbsp;·&nbsp; GoEmotions dataset
  </footer>

</div>

<script>
  const textarea = document.getElementById('txt');
  const charCount = document.getElementById('charCount');
  const history = [];
  const micBtn = document.getElementById('micBtn');
  const translateBtn = document.getElementById('translateBtn');
  const languageSelect = document.getElementById('languageSelect');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;

  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length;
  });

  document.getElementById("translateBtn").addEventListener("click", async () => {
    const text = textarea.value;
    const language = languageSelect.value;
    const errMsg = document.getElementById('errMsg');

    errMsg.classList.remove('visible');

    try {
      const response = await fetch("/translate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text, language })
      });

      const data = await response.json();

      if (data.error) throw new Error(data.error);

      if (data.translated) {
        textarea.value = data.translated;
        charCount.textContent = textarea.value.length;
      }
    } catch (e) {
      errMsg.textContent = 'Translation failed. Please try again.';
      errMsg.classList.add('visible');
    }
  });

  function startSpeechInput() {
    const errMsg = document.getElementById('errMsg');
    errMsg.classList.remove('visible');

    if (!SpeechRecognition) {
      errMsg.textContent = 'Speech recognition is not supported in this browser.';
      errMsg.classList.add('visible');
      return;
    }

    if (!recognition) {
      recognition = new SpeechRecognition();
      recognition.lang = 'en-IN';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        micBtn.classList.add('active');
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript || '';
        textarea.value = textarea.value
          ? textarea.value.trim() + ' ' + transcript
          : transcript;
        charCount.textContent = textarea.value.length;
      };

      recognition.onerror = () => {
        errMsg.textContent = 'Could not capture speech. Please try again.';
        errMsg.classList.add('visible');
      };

      recognition.onend = () => {
        micBtn.classList.remove('active');
      };
    }

    recognition.start();
  }

  async function analyze() {
    const text = textarea.value.trim();
    const language = languageSelect.value;
    const btn = document.getElementById('analyzeBtn');
    const errMsg = document.getElementById('errMsg');

    errMsg.classList.remove('visible');

    if (!text) {
      errMsg.textContent = 'Please enter some text before analysing.';
      errMsg.classList.add('visible');
      return;
    }
    if (text.length < 3) {
      errMsg.textContent = 'Please enter at least a few words for meaningful analysis.';
      errMsg.classList.add('visible');
      return;
    }

    btn.classList.add('loading');

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language }),
      });
      const data = await res.json();

      if (data.error) throw new Error(data.error);

      showResult(data, text);
      addHistory(data, text);
    } catch (e) {
      errMsg.textContent = 'Something went wrong. Is the Flask server running?';
      errMsg.classList.add('visible');
    } finally {
      btn.classList.remove('loading');
    }
  }

  function showResult(data, text) {
    const result       = document.getElementById('result');
    const header       = document.getElementById('resultHeader');
    const icon         = document.getElementById('resultIcon');
    const label        = document.getElementById('resultLabel');
    const tag          = document.getElementById('resultTag');
    const confValue    = document.getElementById('confValue');
    const barTrack     = document.getElementById('barTrack');
    const barFill      = document.getElementById('barFill');
    const interp       = document.getElementById('interpretation');

    const sentimentClass = data.sentiment_class || 'neutral';

    result.classList.remove('visible');
    void result.offsetWidth;

    header.className = 'result-header ' + sentimentClass;
    barTrack.className = 'bar-track ' + sentimentClass;

    icon.textContent =
      sentimentClass === 'positive' ? '🌿' :
      sentimentClass === 'negative' ? '🌧️' : '⛅';
    label.textContent = data.label || data.sentiment;
    tag.textContent = sentimentClass.toUpperCase();
    confValue.textContent = data.confidence + '%';
    interp.innerHTML =
      "<strong>Emotion:</strong> " + data.emotion +
      "<br/><br/>" +
      data.interpretation +
      "<br/><br/><em>" + data.suggestion + "</em>";

    barFill.style.width = '0%';
    setTimeout(() => { barFill.style.width = data.confidence + '%'; }, 50);

    result.classList.add('visible');
  }

  function addHistory(data, text) {
    history.unshift({ data, text });
    if (history.length > 5) history.pop();

    const section = document.getElementById('historySection');
    const list = document.getElementById('historyList');
    section.style.display = 'block';
    list.innerHTML = '';

    history.forEach((h) => {
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML = `
        <div class="history-dot ${(h.data.sentiment_class || 'neutral')}"></div>
        <div class="history-text">${h.text.substring(0, 80)}${h.text.length > 80 ? '…' : ''}</div>
        <div class="history-conf">${h.data.confidence}%</div>
      `;
      item.onclick = () => {
        textarea.value = h.text;
        charCount.textContent = h.text.length;
        showResult(h.data, h.text);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      };
      list.appendChild(item);
    });
  }

  textarea.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
  });
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/history", methods=["GET"])
def history_route():
    return jsonify(load_history())


@app.route("/translate", methods=["POST"])
def translate_text_route():
    data = request.json
    text = data.get("text", "")
    lang = data.get("language", "en")

    transliterated = transliterate_text(text, lang)

    return jsonify({
        "translated": transliterated
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    language = data.get("language", "en").strip().lower()

    if language not in {"en", "hi", "mr"}:
        language = "en"

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = predict_sentiment(text, language)
        save_history({
            "text": text,
            "sentiment": result.get("sentiment_class", "neutral"),
            "emotion": result.get("emotion", "mixed"),
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "language": result.get("language", language),
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🧠 MindRead — Mental Health Sentiment Analyzer")
    print("   Open: http://127.0.0.1:5000")
    app.run(debug=True)

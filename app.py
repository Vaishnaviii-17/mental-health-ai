"""
Therapy-Style AI Mental Health Journal — Flask Web App
=====================================================
Run:
    pip install flask deep-translator indic-transliteration
    python app.py

Then open: http://127.0.0.1:5000
"""

import json
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pickle
import re
from datetime import datetime

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

import numpy as np
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, render_template_string, request
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from scipy.sparse import hstack

# ─── Constants ───────────────────────────────────────────────
MODEL_DIR = "mental_health_model"
JOURNAL_FILE = "journal_history.json"
CLASS_LABELS = ["positive", "negative", "neutral"]

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
    "sad": {
        "weight": 2,
        "keywords": [
            "sad", "hopeless", "crying", "empty", "lonely", "down",
            "meaningless", "worthless", "numb", "low", "hurt",
        ],
    },
    "anxious": {
        "weight": 2,
        "keywords": [
            "anxious", "nervous", "worried", "overthinking", "panic",
            "stressed", "stress", "overwhelmed", "fear", "uneasy",
        ],
    },
    "happy": {
        "weight": 2,
        "keywords": [
            "happy", "excited", "great", "grateful", "joy", "relieved",
            "good", "calm", "proud", "hopeful",
        ],
    },
    "angry": {
        "weight": 2,
        "keywords": [
            "angry", "frustrated", "mad", "annoyed", "irritated", "furious",
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

THERAPY_RESPONSES = {
    "sad": "I'm really sorry you're feeling this way. Would you like to talk about what made today difficult?",
    "anxious": "It seems like your mind is racing. Try focusing on one small step you can control.",
    "happy": "It's great to see positive energy today. What made today better?",
    "neutral": "Today seems calm and steady. Even quiet days are important for balance.",
    "angry": "It sounds like something frustrated you. Taking a short pause might help.",
    "fatigue": "You sound drained. Resting and reducing pressure for a while could really help.",
    "mixed": "Thank you for sharing this. It sounds like there may be a lot happening at once.",
}

REFLECTION_QUESTIONS = {
    "sad": "What do you think contributed most to this feeling today?",
    "anxious": "Is there something specific worrying you right now?",
    "happy": "What is one thing you want to remember from today?",
    "neutral": "Was there any small moment that stood out?",
    "angry": "What triggered this feeling?",
    "fatigue": "What has been draining most of your energy recently?",
    "mixed": "Would you like to write a little more about what feels most present right now?",
}


# ─── Model Loading ───────────────────────────────────────────
def load_artifacts():
    artifacts = {}
    for name in ("model", "tfidf_word", "tfidf_char", "metadata"):
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "rb") as fh:
            artifacts[name] = pickle.load(fh)
    return artifacts


ARTIFACTS = load_artifacts()
print(f"✅ Model loaded — accuracy: {ARTIFACTS['metadata']['accuracy'] * 100:.2f}%")


# ─── Text Processing ─────────────────────────────────────────
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

    strong_neutral_phrases = [
        "nothing stood out",
        "no strong feelings",
        "just routine",
        "felt normal",
        "average day",
        "nothing much happened"
    ]

    return any(p in text for p in strong_neutral_phrases)


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

def detect_journal_type(sentiment, emotion, text):
    text = text.lower()

    if sentiment == "positive":
        if any(word in text for word in ["grateful", "thankful", "blessed", "appreciate"]):
            return "gratitude"
        return "positive_reflection"

    if sentiment == "negative":
        if any(word in text for word in ["angry", "frustrated", "hate", "annoyed"]):
            return "venting"
        if emotion in {"sad", "tired", "anxious"}:
            return "emotional_processing"
        return "venting"

    if sentiment == "neutral":
        return "daily_log"

    return "reflection"

def get_intensity(confidence):
    if confidence >= 85:
        return "strong"
    if confidence >= 65:
        return "moderate"
    return "slight"


def generate_interpretation(sentiment_class, intensity, emotion):
    if sentiment_class == "negative":
        if emotion == "fatigue":
            return f"You seem mentally tired or drained, with a {intensity} negative tone."
        if emotion == "sad":
            return f"Your writing reflects a {intensity} sadness that may feel emotionally heavy."
        if emotion == "anxious":
            return f"There are signs of a {intensity} anxious state, with stress or worry showing through."
        if emotion == "angry":
            return f"Your words carry a {intensity} frustration that may come from something upsetting or unfair."
        if emotion == "mixed":
            return f"Your message suggests a {intensity} difficult emotional state with overlapping feelings."
        return f"The text shows a {intensity} negative tone that may reflect emotional strain."

    if sentiment_class == "positive":
        if emotion == "happy":
            return f"Your writing reflects a {intensity} positive state with signs of comfort, joy, or relief."
        return f"The overall tone feels {intensity}ly positive and emotionally supportive."

    if emotion == "neutral":
        return "Your day appears steady and balanced, though it may feel routine or uneventful."
    return "Your emotional state appears fairly balanced without very strong positive or negative signals."


def generate_suggestion(sentiment_class, emotion):
    if sentiment_class == "negative":
        if emotion == "fatigue":
            return "Try rest, hydration, and reducing one demand at a time."
        if emotion == "anxious":
            return "Try slow breathing, grounding, or focusing on one manageable next step."
        if emotion == "sad":
            return "Journaling or reaching out to someone you trust may help lighten the load."
        if emotion == "angry":
            return "A short pause, a walk, or writing the trigger down may help release tension."
        return "Try taking breaks, journaling, or speaking with someone you trust."

    if sentiment_class == "positive":
        return "Keep engaging in the people, habits, or activities that are supporting your well-being."

    return "Consider adding one small meaningful activity to bring variety or comfort to your day."


def generate_therapy_response(emotion, sentiment):
    if sentiment == "negative":
        if emotion == "fatigue":
            return "You sound really drained. Was it more mental exhaustion or physical?"
        if emotion == "sad":
            return "That sounds emotionally heavy. Do you feel this has been building over time?"
        if emotion == "anxious":
            return "It seems like your mind is quite active right now. Is there something specific causing this?"
        if emotion == "angry":
            return "That frustration feels real. What triggered it today?"
        return "It seems like something is weighing on you. Do you want to explore it a bit more?"

    if sentiment == "positive":
        return "It's nice to see some positive energy. What made today feel better?"

    if sentiment == "neutral":
        return "Today seems calm or routine. Did anything small stand out to you?"

    return "Thank you for sharing. Would you like to expand on that?"


import random

def generate_reflection_question(emotion):
    questions = {
        "sad": [
            "What do you think contributed most to this feeling?",
            "Has something like this been on your mind for a while?"
        ],
        "anxious": [
            "Is there a specific thought looping in your mind?",
            "What feels most uncertain right now?"
        ],
        "happy": [
            "What is one thing you want to remember from today?",
            "What made today better than usual?"
        ],
        "neutral": [
            "Was there any small moment that stood out?",
            "Did anything feel slightly different today?"
        ],
        "fatigue": [
            "What has been draining your energy the most?",
            "Have you been getting enough rest lately?"
        ]
    }

    return random.choice(questions.get(emotion, ["Would you like to write more about this?"]))


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

    if emotion in {"fatigue", "sad", "anxious", "angry"} and sentiment_class == "positive":
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

    cleaned = clean_text(translated_text)

    if not cleaned:
        journal_message = f"{therapy_response} {reflection_question}"
        
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
            "therapy_response": "Thank you for sharing your thoughts.",
            "reflection_question": "Would you like to write more about this?",
            "journal_message": "Thank you for sharing.",
        }

    sentiment_class, confidence = predict_model_sentiment(cleaned)
    emotion = detect_emotion(translated_text)
    journal_type = detect_journal_type(sentiment_class, emotion, translated_text)
    sentiment_class, confidence = apply_rule_overrides(translated_text, sentiment_class, confidence, emotion)

    intensity = get_intensity(confidence)
    sentiment = format_sentiment(sentiment_class, intensity)
    interpretation = generate_interpretation(sentiment_class, intensity, emotion)
    suggestion = generate_suggestion(sentiment_class, emotion)
    therapy_response = generate_therapy_response(emotion, sentiment_class)
    reflection_question = generate_reflection_question(emotion)
    journal_message = f"{therapy_response} {reflection_question}"
    

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
        "therapy_response": therapy_response,
        "reflection_question": reflection_question,
        "journal_type": journal_type,
        "journal_message": journal_message  
    }


# ─── Journal Storage ─────────────────────────────────────────
def ensure_journal_file():
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w", encoding="utf-8") as fh:
            json.dump([], fh)


def load_history():
    ensure_journal_file()
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def write_history(entries):
    with open(JOURNAL_FILE, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)


def save_entry(text, sentiment, emotion, confidence, suggestion):
    entries = load_history()
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": text,
        "sentiment": sentiment,
        "emotion": emotion,
        "confidence": confidence,
        "suggestion": suggestion,
    }
    entries.append(entry)
    write_history(entries)
    return entry


def generate_trend(entries):
    recent = entries[-5:]
    if not recent:
        return "Start journaling to build a picture of your recent mood patterns."

    counts = {}
    for entry in recent:
        emotion = entry.get("emotion", "mixed")
        counts[emotion] = counts.get(emotion, 0) + 1

    dominant_emotion = max(counts, key=counts.get)
    messages = {
        "sad": "You have been feeling more sad recently.",
        "anxious": "You have been feeling slightly anxious recently.",
        "happy": "You have been showing more positive energy recently.",
        "neutral": "Your recent entries seem fairly steady and balanced.",
        "angry": "Frustration seems to have appeared in your recent entries.",
        "fatigue": "You have sounded mentally tired in several recent entries.",
        "mixed": "Your recent entries show a mix of different emotions.",
    }
    return messages.get(dominant_emotion, "Your recent entries show a mix of different emotions.")


# ─── Flask App ───────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Therapy-Style AI Mental Health Journal</title>
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

  .wrap { width: 100%; max-width: 760px; position: relative; z-index: 1; }

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
    padding: 28px;
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
    min-height: 160px;
    line-height: 1.7;
    padding: 16px 18px;
    resize: vertical;
  }

  textarea::placeholder { color: var(--muted); }
  textarea:focus,
  select:focus { border-color: var(--accent); }

  .action-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
  }

  .tool-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 10px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: border-color .2s, opacity .2s;
  }
  .tool-btn:hover { border-color: var(--accent); }
  .tool-btn.active {
    border-color: var(--accent);
    color: var(--accent);
  }

  .primary-btn {
    margin-left: auto;
    background: var(--accent);
    color: white;
    border-color: transparent;
  }

  .char-count {
    text-align: right;
    font-size: 12px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-top: 10px;
  }

  .result {
    display: none;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    overflow: hidden;
    animation: slideUp .35s ease;
  }
  .result.visible { display: block; }

  .result-header {
    padding: 24px 28px;
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .result-header.positive { background: var(--pos-dim); border-bottom: 1px solid #166534; }
  .result-header.negative { background: var(--neg-dim); border-bottom: 1px solid #991b1b; }
  .result-header.neutral { background: var(--neu-dim); border-bottom: 1px solid #a16207; }

  .result-icon { font-size: 2.2rem; line-height: 1; }
  .result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
  }
  .result-header.positive .result-label { color: var(--pos); }
  .result-header.negative .result-label { color: var(--neg); }
  .result-header.neutral .result-label { color: var(--neu); }

  .result-tag {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .1em;
    padding: 4px 10px;
    border-radius: 20px;
  }
  .result-header.positive .result-tag { background: #166534; color: var(--pos); }
  .result-header.negative .result-tag { background: #991b1b; color: var(--neg); }
  .result-header.neutral .result-tag { background: #a16207; color: var(--neu); }

  .result-body {
    background: var(--surface);
    padding: 24px 28px;
  }

  .conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .conf-label, .conf-value {
    font-size: 13px;
    font-family: 'DM Mono', monospace;
  }
  .conf-label { color: var(--muted); }

  .bar-track {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
  }
  .positive .bar-fill { background: var(--pos); }
  .negative .bar-fill { background: var(--neg); }
  .neutral .bar-fill { background: var(--neu); }

  .detail-block {
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    line-height: 1.7;
    font-size: 14px;
  }
  .detail-block strong {
    color: var(--text);
    font-weight: 500;
  }

  .history-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .history-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .history-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .history-dot.positive { background: var(--pos); }
  .history-dot.negative { background: var(--neg); }
  .history-dot.neutral { background: var(--neu); }

  .history-date {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    min-width: 120px;
  }

  .history-meta {
    flex: 1;
    font-size: 13px;
    color: var(--muted);
  }

  .trend-box {
    font-size: 14px;
    line-height: 1.7;
    color: var(--muted);
  }

  .error-msg {
    display: none;
    background: #2d1515;
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 14px;
    color: var(--neg);
  }
  .error-msg.visible { display: block; }

  footer {
    margin-top: 24px;
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: .08em;
  }
  footer span { color: var(--accent); }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo"><div class="logo-dot"></div>Therapy Journal AI<div class="logo-dot"></div></div>
    <h1>A gentle space to<br/><em>reflect</em> and write</h1>
    <p class="subtitle">Speak or type your thoughts, translate if needed, and explore supportive reflections on how you feel.</p>
  </header>

  <div class="card">
    <label for="languageSelect">Language</label>
    <select id="languageSelect">
      <option value="en">English</option>
      <option value="hi">Hindi</option>
      <option value="mr">Marathi</option>
    </select>

    <label for="txt">Journal entry</label>
    <textarea id="txt" placeholder="Write about your day, your thoughts, or how you're feeling right now..."></textarea>

    <div class="action-row">
      <button class="tool-btn" id="micBtn" type="button" onclick="startSpeechInput()">🎙 Speak</button>
      <button class="tool-btn" id="translateBtn" type="button">🌐 Translate</button>
      <button class="tool-btn" id="saveBtn" type="button" onclick="saveJournalEntry()">Save Journal Entry</button>
      <button class="tool-btn primary-btn" id="analyzeBtn" type="button" onclick="analyze()">Analyze</button>
    </div>

    <div class="char-count"><span id="charCount">0</span> characters</div>
    <div class="error-msg" id="errMsg"></div>
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

      <div class="detail-block" id="emotionBlock"></div>
      <div class="detail-block" id="interpretationBlock"></div>
      <div class="detail-block" id="therapyBlock"></div>
      <div class="detail-block" id="questionBlock"></div>
      <div class="detail-block" id="trendBlock"></div>
      <div style="margin-top:15px;text-align:center;">
  <button onclick="openChat()" style="
    background:#4CAF50;
    border:none;
    padding:10px 16px;
    border-radius:8px;
    color:white;
    cursor:pointer;
  ">
    💬 Would you like to talk more?
  </button>
</div>
    </div>
  </div>

  <div class="card">
    <div class="history-title">Your Journal History</div>
    <div class="history-list" id="journalHistoryList"></div>
  </div>

  <footer>
    Model accuracy <span>63.61%</span> · LinearSVC + TF-IDF · GoEmotions dataset
  </footer>
</div>

<script>
  const textarea = document.getElementById('txt');
  const charCount = document.getElementById('charCount');
  const languageSelect = document.getElementById('languageSelect');
  const micBtn = document.getElementById('micBtn');
  const translateBtn = document.getElementById('translateBtn');
  const saveBtn = document.getElementById('saveBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const errMsg = document.getElementById('errMsg');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  let recognition = null;
  let lastAnalysis = null;

  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length;
  });

  window.onclick = function(e) {
  const modal = document.getElementById("chatModal");
  if (e.target === modal) {
    modal.style.display = "none";
  }
}
   function openChat() {
  document.getElementById("chatModal").style.display = "flex";
}

function closeChat() {
  document.getElementById("chatModal").style.display = "none";
}

function appendMessage(sender, text) {
  const box = document.getElementById("chatMessages");
  const msg = document.createElement("div");

  msg.style.marginBottom = "10px";
  msg.style.padding = "8px 10px";
  msg.style.borderRadius = "8px";
  msg.style.fontSize = "14px";
  msg.style.lineHeight = "1.4";

  if (sender === "You") {
    msg.style.background = "#4CAF50";
    msg.style.color = "white";
    msg.style.alignSelf = "flex-end";
    msg.style.textAlign = "right";
  } else {
    msg.style.background = "#2a2f45";
    msg.style.color = "#e8eaf0";  // 👈 FIXED TEXT COLOR
  }

  msg.innerHTML = text;
  box.appendChild(msg);

  box.scrollTop = box.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const box = document.getElementById("chatMessages");

  const msg = input.value.trim();
  if (!msg) return;

  // USER MESSAGE
  box.innerHTML += `
    <div style="text-align:right; margin:8px;">
      <span style="background:#4ade80; padding:6px 10px; border-radius:10px;">
        ${msg}
      </span>
    </div>
  `;

  input.value = "";

  // AI typing
  const typingId = "typing-" + Date.now();
  box.innerHTML += `
    <div id="${typingId}" style="text-align:left; margin:8px;">
      <span style="background:#2a2f45; padding:6px 10px; border-radius:10px;">
        Typing...
      </span>
    </div>
  `;

  box.scrollTop = box.scrollHeight;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: msg,
        emotion: lastAnalysis?.emotion,
        sentiment: lastAnalysis?.sentiment_class
      })
    });

    const data = await res.json();

    // replace typing with real reply
    document.getElementById(typingId).innerHTML = `
      <span style="background:#2a2f45; padding:6px 10px; border-radius:10px;">
        ${data.reply}
      </span>
    `;

  } catch (err) {
    document.getElementById(typingId).innerHTML = `
      <span style="background:#2a2f45; padding:6px 10px; border-radius:10px;">
        I'm here for you. Tell me more.
      </span>
    `;
  }

  box.scrollTop = box.scrollHeight;
}

  function showError(message) {
    errMsg.textContent = message;
    errMsg.classList.add('visible');
  }

  function clearError() {
    errMsg.classList.remove('visible');
  }

  function updateCharCount() {
    charCount.textContent = textarea.value.length;
  }

  async function refreshHistory() {
    try {
      const res = await fetch('/history');
      const data = await res.json();
      const entries = data.entries || [];
      const trend = data.trend || '';

      const list = document.getElementById('journalHistoryList');
      list.innerHTML = '';

      if (!entries.length) {
        list.innerHTML = '<div class="history-meta">No journal entries saved yet.</div>';
      } else {
        entries.forEach((entry) => {
          const item = document.createElement('div');
          item.className = 'history-item';
          item.innerHTML = `
            <div class="history-dot ${entry.sentiment_class || 'neutral'}"></div>
            <div class="history-date">${entry.date}</div>
            <div class="history-meta">${entry.emotion} | ${entry.sentiment}</div>
          `;
          list.appendChild(item);
        });
      }

      if (trend && lastAnalysis) {
        document.getElementById('trendBlock').innerHTML = '<strong>Trend Insight:</strong><br/>' + trend;
      }
    } catch (e) {
      console.error('Failed to load history');
    }
  }

window.analyze = async function () {
  const textarea = document.getElementById('txt');
  const text = textarea.value.trim();

  const btn = document.getElementById('analyzeBtn');
  const errMsg = document.getElementById('errMsg');

  errMsg.classList.remove('visible');

  if (!text) {
    errMsg.textContent = 'Please enter some text before analysing.';
    errMsg.classList.add('visible');
    return;
  }

  btn.classList.add('loading');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
  text: text,
  language: document.getElementById('languageSelect').value
}),
    });

    const data = await res.json();

    if (data.error) throw new Error(data.error);

    // ✅ THIS IS THE IMPORTANT PART
    showResult(data, text);

    lastAnalysis = data;

  } catch (e) {
    errMsg.textContent = 'Something went wrong. Is the Flask server running?';
    errMsg.classList.add('visible');
  } finally {
    btn.classList.remove('loading');
  }
};

  async function saveJournalEntry() {
    clearError();

    if (!lastAnalysis) {
      showError('Please analyze an entry before saving it.');
      return;
    }

    saveBtn.disabled = true;

    try {
      const res = await fetch('/save-entry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: lastAnalysis.original_text,
          sentiment: lastAnalysis.sentiment,
          sentiment_class: lastAnalysis.sentiment_class,
          emotion: lastAnalysis.emotion,
          confidence: lastAnalysis.confidence,
          suggestion: lastAnalysis.suggestion
        }),
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      await refreshHistory();
    } catch (e) {
      showError('Could not save journal entry. Please try again.');
    } finally {
      saveBtn.disabled = false;
    }
  }

  document.getElementById('translateBtn').addEventListener('click', async () => {
    const text = textarea.value.trim();
    const language = languageSelect.value;

    clearError();

    if (!text) {
      showError('Please enter some text before translating.');
      return;
    }

    if (!['hi', 'mr'].includes(language)) {
      showError('Please select Hindi or Marathi to transliterate.');
      return;
    }

    translateBtn.disabled = true;

    try {
      const response = await fetch('/translate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text, language })
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      if (data.translated) {
        textarea.value = data.translated;
        updateCharCount();
      }
    } catch (e) {
      showError('Translation failed. Please try again.');
    } finally {
      translateBtn.disabled = false;
    }
  });

  function startSpeechInput() {
    clearError();

    if (!SpeechRecognition) {
      showError('Speech recognition is not supported in this browser.');
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
        updateCharCount();
      };

      recognition.onerror = () => {
        showError('Could not capture speech. Please try again.');
      };

      recognition.onend = () => {
        micBtn.classList.remove('active');
      };
    }

    recognition.start();
  }

  function showResult(data) {
    const result = document.getElementById('result');
    const header = document.getElementById('resultHeader');
    const icon = document.getElementById('resultIcon');
    const label = document.getElementById('resultLabel');
    const tag = document.getElementById('resultTag');
    const confValue = document.getElementById('confValue');
    const barTrack = document.getElementById('barTrack');
    const barFill = document.getElementById('barFill');

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

    document.getElementById('emotionBlock').innerHTML =
      '<strong>Emotion:</strong><br/>' + data.emotion;

    document.getElementById('interpretationBlock').innerHTML =
      '<strong>Interpretation:</strong><br/>' + data.interpretation;

document.getElementById('therapyBlock').innerHTML =
  '<strong>Journal Reflection:</strong><br/>' + data.journal_message || "No reflection available.";

document.getElementById('questionBlock').innerHTML = '';

document.getElementById('trendBlock').innerHTML =
  '<strong>Trend Insight:</strong><br/>' + (data.trend_insight || '');

    barFill.style.width = '0%';
    setTimeout(() => { barFill.style.width = data.confidence + '%'; }, 50);

    result.classList.add('visible');
  }

  textarea.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
  });

  refreshHistory();
</script>
<div id="chatModal" style="
  display:none;
  position:fixed;
  top:0;
  left:0;
  width:100%;
  height:100%;
  background:rgba(0,0,0,0.6);
  justify-content:center;
  align-items:center;
  z-index:999;
">

  <div style="
    width:420px;
    height:500px;
    background:#1e1e2f;
    border-radius:12px;
    box-shadow:0 20px 60px rgba(0,0,0,0.6);
    display:flex;
    flex-direction:column;
    overflow:hidden;
  ">

<div style="padding:12px; background:#2a2f45; color:#e8eaf0; border-radius:10px 10px 0 0;">
  AI Listener
  <span onclick="closeChat()" style="float:right; cursor:pointer;">✖</span>
</div>

 <div id="chatMessages" style="
  flex:1;
  padding:10px;
  overflow-y:auto;
  font-size:14px;
  display:flex;
  flex-direction:column;
  gap:10px;
">
</div>

  <div style="display:flex;">
<input id="chatInput" type="text" placeholder="Share your thoughts..." style="
  flex:1;
  border:none;
  padding:8px;
  outline:none;
  background:#0d0f14;
  color:white;
">
    <button onclick="sendMessage()" style="
      background:#4CAF50;
      border:none;
      color:white;
      padding:8px;
    ">Send</button>
  </div>

</div>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    language = data.get("language", "en").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if language not in {"hi", "mr"}:
        return jsonify({"error": "Translation supports only Hindi or Marathi"}), 400

    return jsonify({"translated": transliterate_text(text, language)})


@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    language = data.get("language", "en").strip().lower()

    if language not in {"en", "hi", "mr"}:
        language = "en"

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = predict_sentiment(text, language)
        result["trend_insight"] = generate_trend(load_history())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_alias():
    return analyze_route()


@app.route("/save-entry", methods=["POST"])
def save_entry_route():
    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()
    sentiment = data.get("sentiment", "").strip()
    sentiment_class = data.get("sentiment_class", "neutral").strip()
    emotion = data.get("emotion", "mixed").strip()
    confidence = float(data.get("confidence", 0.0))
    suggestion = data.get("suggestion", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    entry = save_entry(text, sentiment, emotion, confidence, suggestion)
    entry["sentiment_class"] = sentiment_class
    entries = load_history()
    if entries:
      entries[-1]["sentiment_class"] = sentiment_class
      write_history(entries)

    return jsonify({
        "saved": True,
        "entry": entry,
        "trend": generate_trend(load_history())
    })


@app.route("/history", methods=["GET"])
def history_route():
    entries = load_history()
    last_ten = list(reversed(entries[-10:]))
    return jsonify({
        "entries": last_ten,
        "trend": generate_trend(entries)
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")

    if not user_msg:
        return jsonify({"reply": "Tell me what's on your mind."})

    try:
        prompt = f"""
You are a warm, empathetic mental health journaling companion.

Rules:
- Talk like a human, not a robot
- Be supportive and calm
- Ask thoughtful follow-up questions
- Do NOT repeat yourself
- Keep responses short but meaningful

User: {user_msg}
AI:
"""

        response = model.generate_content(prompt)

        reply = response.text.strip()

        return jsonify({"reply": reply})

    except Exception as e:
        print(e)
        return jsonify({"reply": "I'm here for you. Tell me more."})

if __name__ == "__main__":
    ensure_journal_file()
    print("🧠 Therapy-Style AI Mental Health Journal")
    print("   Open: http://127.0.0.1:5000")
    app.run(debug=True)

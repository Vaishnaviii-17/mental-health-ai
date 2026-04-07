"""
Mental Health Sentiment Analyzer
=================================
Dataset  : GoEmotions (211K Reddit comments, 28 emotion labels)
Model    : LinearSVC with TF-IDF (word + char n-grams)
Task     : Sentiment classification — Positive vs Negative vs Neutral

Usage
-----
# Train:
    python mental_health_sentiment_train.py

# Predict (after training):
    python mental_health_sentiment_train.py --predict
"""

import argparse
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

EMOTION_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

POSITIVE_EMOTIONS = [
    "admiration", "amusement", "approval", "caring", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
]

NEGATIVE_EMOTIONS = [
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
]

MODEL_DIR = "mental_health_model"
CLASS_LABELS = ["positive", "negative", "neutral"]


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_primary_emotion(row) -> str:
    """Return the first active emotion label for a row (single-label assumed)."""
    for col in EMOTION_COLS:
        if row[col] == 1:
            return col
    return "none"


def map_sentiment(emotion: str) -> str:
    if emotion in POSITIVE_EMOTIONS:
        return "positive"
    elif emotion in NEGATIVE_EMOTIONS:
        return "negative"
    else:
        return "neutral"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\[NAME\]", "someone", text)
    text = re.sub(r"[^a-z\s'!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_confidence(scores: np.ndarray) -> np.ndarray:
    """
    Convert LinearSVC margins into stable pseudo-probabilities.
    """
    scores = np.asarray(scores, dtype=float)

    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])

    clipped = np.clip(scores, -8.0, 8.0)
    clipped = clipped - clipped.max(axis=1, keepdims=True)
    exp_scores = np.exp(clipped)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs


# ─── DATA LOADING & PREPROCESSING ────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    print(f"[1/5] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Total records: {len(df):,}")

    print("[2/5] Preprocessing & filtering...")
    df["primary_emotion"] = df[EMOTION_COLS].apply(get_primary_emotion, axis=1)
    df["sentiment"] = df["primary_emotion"].map(map_sentiment)

    df_clean = df[
        (df["example_very_unclear"] == False)
        & (df[EMOTION_COLS].sum(axis=1) == 1)
    ].copy()

    df_clean["clean_text"] = df_clean["text"].apply(clean_text)
    df_clean = df_clean[df_clean["clean_text"].str.len() > 3].reset_index(drop=True)

    print(f"      Records after filtering : {len(df_clean):,}")
    print(f"      Positive                : {(df_clean['sentiment'] == 'positive').sum():,}")
    print(f"      Negative                : {(df_clean['sentiment'] == 'negative').sum():,}")
    print(f"      Neutral                 : {(df_clean['sentiment'] == 'neutral').sum():,}")
    return df_clean


# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def build_features(X_train, X_test):
    """Return (X_tr, X_te, tfidf_word, tfidf_char) using combined TF-IDF."""
    print("[4/5] Extracting TF-IDF features (word + char n-grams)...")

    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=200_000,
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
    )
    tfidf_char = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=100_000,
        sublinear_tf=True,
        min_df=5,
        analyzer="char_wb",
    )

    X_tr = hstack([tfidf_word.fit_transform(X_train), tfidf_char.fit_transform(X_train)])
    X_te = hstack([tfidf_word.transform(X_test), tfidf_char.transform(X_test)])

    print(f"      Feature dimensions: {X_tr.shape[1]:,}")
    return X_tr, X_te, tfidf_word, tfidf_char


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(csv_path: str = "dataset.csv"):
    df = load_and_preprocess(csv_path)

    print("[3/5] Train / test split (80 / 20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    X_tr, X_te, tfidf_word, tfidf_char = build_features(X_train, X_test)

    print("[5/5] Training LinearSVC...")
    model = LinearSVC(C=1.0, max_iter=4000, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted F1      : {f1_w:.4f}")
    print(f"  Macro F1         : {f1_m:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS))
    print(f"\n{classification_report(y_test, y_pred, labels=CLASS_LABELS)}")
    print(f"  {'✅ TARGET ACHIEVED' if acc >= 0.80 else '❌ TARGET NOT MET'}: accuracy = {acc:.4f}")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    metadata = {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "positive_emotions": POSITIVE_EMOTIONS,
        "negative_emotions": NEGATIVE_EMOTIONS,
        "classes": CLASS_LABELS,
    }

    for name, obj in [
        ("model", model),
        ("tfidf_word", tfidf_word),
        ("tfidf_char", tfidf_char),
        ("metadata", metadata),
    ]:
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as fh:
            pickle.dump(obj, fh)

    print(f"\n  Saved: {MODEL_DIR}/{{model, tfidf_word, tfidf_char, metadata}}.pkl")
    return model, tfidf_word, tfidf_char


# ─── INFERENCE ───────────────────────────────────────────────────────────────

def load_model():
    """Load saved model artefacts from MODEL_DIR."""
    artefacts = {}
    for name in ("model", "tfidf_word", "tfidf_char", "metadata"):
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(path, "rb") as fh:
            artefacts[name] = pickle.load(fh)
    return artefacts


def predict(texts: list[str], artefacts: dict) -> list[dict]:
    """
    Predict mental health sentiment for a list of raw text strings.

    Returns
    -------
    list of dicts: [{"text": ..., "sentiment": "...", "confidence": float}, ...]
    """
    model = artefacts["model"]
    tfidf_word = artefacts["tfidf_word"]
    tfidf_char = artefacts["tfidf_char"]

    cleaned = [clean_text(t) for t in texts]
    X = hstack([tfidf_word.transform(cleaned), tfidf_char.transform(cleaned)])

    predictions = model.predict(X)
    probs = normalize_confidence(model.decision_function(X))
    classes = list(getattr(model, "classes_", CLASS_LABELS))

    results = []
    for text, pred, prob_row in zip(texts, predictions, probs):
        pred_idx = classes.index(pred)
        results.append({
            "text": text,
            "sentiment": pred,
            "confidence": round(float(prob_row[pred_idx]) * 100, 2),
        })
    return results


# ─── DEMO INTERACTIVE PREDICT ────────────────────────────────────────────────

def interactive_predict():
    print("\nLoading saved model...")
    artefacts = load_model()
    meta = artefacts["metadata"]
    print(f"Model accuracy: {meta['accuracy']*100:.2f}%\n")

    demo_texts = [
        "I feel so hopeless, nothing ever gets better.",
        "Today was amazing! I got the job I always wanted.",
        "I'm really anxious about the upcoming exam.",
        "Feeling grateful for all the support from my friends.",
        "Everything is falling apart and I don't know what to do.",
        "Had a great workout and feeling energised!",
        "I can't stop crying and I don't even know why.",
        "Life feels meaningless lately.",
        "Today was routine and nothing really stood out.",
    ]

    results = predict(demo_texts, artefacts)

    print("=" * 60)
    print("  DEMO PREDICTIONS")
    print("=" * 60)
    for r in results:
        if r["sentiment"] == "positive":
            emoji = "😊"
        elif r["sentiment"] == "negative":
            emoji = "😔"
        else:
            emoji = "😐"
        print(f"  {emoji} [{r['sentiment'].upper():8s}] (conf: {r['confidence']:.2f}%) {r['text'][:70]}")
    print("=" * 60)


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mental Health Sentiment Analyzer")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run demo predictions using saved model (must train first)",
    )
    parser.add_argument(
        "--data",
        default="dataset.csv",
        help="Path to GoEmotions CSV dataset (default: dataset.csv)",
    )
    args = parser.parse_args()

    if args.predict:
        interactive_predict()
    else:
        train(csv_path=args.data)
        print("\nRunning demo predictions on sample texts...")
        interactive_predict()

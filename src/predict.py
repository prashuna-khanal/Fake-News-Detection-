"""
predict.py
----------
CLI interface for the Fake News Detection model.

Usage:
  python src/predict.py                  -> Interactive mode (type/paste news)
  python src/predict.py --file news.txt  -> Batch mode (one article per line)
  python src/predict.py --demo           -> Run with built-in demo examples
"""

import os
import re
import sys
import pickle
import argparse
import nltk
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack

nltk.download("stopwords", quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STOP_WORDS = set(stopwords.words("english"))
NEGATIONS = {"no", "not", "never"}

# Refined indicators (Must match preprocess.py)
FAKE_WORDS = [
    "scam", "hoax", "clickbait", "viral", "whatsapp", "viber", 
    "telegram", "shocking", "prize", "gift", "winning", "unfounded",
    "fabricated", "circulating"
]

FAKE_PHRASES = [
    "viral message", "whatsapp message", "video circulating",
    "completely false", "fabricated", "unfounded", "conspiracy theory",
    "free gold", "free house", "free money"
]

BANNER = """
+--------------------------------------------------------------+
|  AI-Based Fake News Detector - Nepal's Digital News Ecosystem |
|  Model: Random Forest Classifier  |  Features: TF-IDF        |
+--------------------------------------------------------------+
"""

DEMO_EXAMPLES = [
    {
        "text": "Nepal Government Announces New Budget for Fiscal Year 2081/82. "
                "The Government of Nepal has announced the national budget allocating "
                "significant funds for infrastructure development, education, and health sectors. "
                "Finance Minister presented the budget in the Federal Parliament.",
        "expected": "Real"
    },
    {
        "text": "BREAKING: Government to Distribute Free Gold to Every Nepali Family. "
                "According to a viral WhatsApp message circulating widely, the government has "
                "announced a scheme to distribute free gold to every Nepali family. "
                "The message claims families must register within 48 hours. No official confirmation exists.",
        "expected": "Fake"
    },
    {
        "text": "Nepal Electricity Authority Reduces Load Shedding Hours. "
                "Nepal Electricity Authority announced a reduction in load shedding hours "
                "following increased water levels in reservoirs. Consumers in Kathmandu Valley "
                "will now experience only two hours of power cuts per day during peak demand.",
        "expected": "Real"
    },
    {
        "text": "SHOCKING: COVID Vaccine Contains Microchip to Track Nepali Citizens. "
                "Viral messages on WhatsApp and Facebook claim that COVID-19 vaccines contain "
                "microchips that allow the government to track citizens movements. "
                "Health experts and vaccine manufacturers have repeatedly debunked this conspiracy theory.",
        "expected": "Fake"
    },
    {
        "text": "Pokhara International Airport Receives First International Flight. "
                "Pokhara International Airport received its first scheduled international flight "
                "from Kuala Lumpur, marking a significant milestone for Nepal aviation sector. "
                "The airport is expected to boost tourism in the Gandaki region.",
        "expected": "Real"
    },
]


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, punctuation, digits, but keep negations."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)    # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip()       # collapse spaces
    tokens = [t for t in text.split() if t not in STOP_WORDS or t in NEGATIONS]
    return " ".join(tokens)


def fake_word_count(texts):
    """
    Extracts word counts and phrase presence as extra features.
    Must match preprocess.py logic.
    """
    features = []
    for text in texts:
        t_lower = text.lower()
        # Word counts
        word_counts = [t_lower.count(word) for word in FAKE_WORDS]
        # Phrase flags (binary)
        phrase_flags = [1 if phrase in t_lower else 0 for phrase in FAKE_PHRASES]
        features.append(word_counts + phrase_flags)
    return np.array(features)


def load_model_and_vectorizer():
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    vec_path   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path):
        print("ERROR: Model not found. Run src/train_model.py first.")
        sys.exit(1)
    if not os.path.exists(vec_path):
        print("ERROR: Vectorizer not found. Run src/preprocess.py first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_single(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {"label": "UNKNOWN", "confidence": 0.0, "fake_prob": 0.0, "real_prob": 0.0}
    
    # Extract TF-IDF features
    tfidf_features = vectorizer.transform([cleaned])
    
    # Extract extra features
    extra_features = fake_word_count([text])
    # Boost: Repeat 2x to match training
    extra_boosted = np.tile(extra_features, 2)
    
    # Combine features
    combined_features = hstack([tfidf_features, extra_boosted])
    
    label_idx = model.predict(combined_features)[0]
    proba = model.predict_proba(combined_features)[0]
    
    label = "[REAL]" if label_idx == 1 else "[FAKE]"
    confidence = proba[label_idx] * 100
    return {
        "label": label,
        "label_idx": label_idx,
        "confidence": confidence,
        "fake_prob": proba[0] * 100,
        "real_prob": proba[1] * 100,
    }


def print_result(text_preview: str, result: dict, index: int = None):
    sep = "-" * 62
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{sep}")
    if text_preview:
        preview = text_preview[:120] + "..." if len(text_preview) > 120 else text_preview
        print(f"  {prefix}Input: {preview}")
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['confidence']:.1f}%")
    print(f"  Fake Prob   : {result['fake_prob']:.1f}%  |  Real Prob: {result['real_prob']:.1f}%")

    # Visual confidence bar (ASCII)
    bar_len = 40
    filled = int(result['confidence'] / 100 * bar_len)
    bar = "#" * filled + "." * (bar_len - filled)
    print(f"  Confidence  : [{bar}] {result['confidence']:.1f}%")
    print(sep)


def interactive_mode(model, vectorizer):
    print(BANNER)
    print("  Interactive Mode - Type or paste a news article/headline.")
    print("  Commands: 'quit' or 'exit' to stop, 'demo' to run demo examples.\n")

    count = 0
    while True:
        try:
            print("\nEnter news text (or 'quit'/'demo'):")
            lines = []
            while True:
                line = input("  > ").strip()
                if line.lower() in ("quit", "exit"):
                    print("\nGoodbye!")
                    return
                if line.lower() == "demo":
                    demo_mode(model, vectorizer)
                    break
                if line == "":
                    if lines:
                        break
                    continue
                lines.append(line)
            if not lines:
                continue
            text = " ".join(lines)
            count += 1
            result = predict_single(text, model, vectorizer)
            print_result(text, result, count)
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break


def demo_mode(model, vectorizer):
    print(f"\n{'='*62}")
    print("  DEMO MODE - Running {n} example predictions".format(n=len(DEMO_EXAMPLES)))
    print(f"{'='*62}")
    correct = 0
    for i, ex in enumerate(DEMO_EXAMPLES, 1):
        result = predict_single(ex["text"], model, vectorizer)
        print_result(ex["text"], result, i)
        predicted = "Real" if result["label_idx"] == 1 else "Fake"
        match = "[CORRECT]" if predicted == ex["expected"] else "[WRONG]"
        print(f"  Expected: {ex['expected']}  ->  {match}")
        if predicted == ex["expected"]:
            correct += 1
    print(f"\n  Demo Accuracy: {correct}/{len(DEMO_EXAMPLES)} ({correct/len(DEMO_EXAMPLES)*100:.0f}%)")


def batch_mode(filepath: str, model, vectorizer):
    print(BANNER)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"  Batch Mode - Processing {len(lines)} articles from: {filepath}\n")
    for i, text in enumerate(lines, 1):
        result = predict_single(text, model, vectorizer)
        print_result(text, result, i)


def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detection CLI - Nepal's Digital News Ecosystem"
    )
    parser.add_argument("--file", type=str, help="Path to text file (one article per line)")
    parser.add_argument("--demo", action="store_true", help="Run demo examples")
    args = parser.parse_args()

    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    print("Model loaded successfully!\n")

    if args.demo:
        print(BANNER)
        demo_mode(model, vectorizer)
    elif args.file:
        batch_mode(args.file, model, vectorizer)
    else:
        interactive_mode(model, vectorizer)


if __name__ == "__main__":
    main()

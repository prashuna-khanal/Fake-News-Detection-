"""
preprocess.py (Improved)
------------------------
Preprocesses English news dataset:
- Cleans text (lowercase, removes URLs, punctuation, digits)
- Keeps negations (not, no, never)
- Adds fake-word count features
- Applies TF-IDF vectorization (1-3 grams)
- Splits into train / validation / test sets

Outputs:
  - models/tfidf_vectorizer.pkl
  - data/processed/X_train.pkl, X_val.pkl, X_test.pkl
  - data/processed/y_train.pkl, y_val.pkl, y_test.pkl
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Stopwords
STOP_WORDS = set(stopwords.words("english"))
NEGATIONS = {"no", "not", "never"}

# Fake-related indicators (Refined to be more specific to misinformation)
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

# ── Text Cleaning ──────────────────────────────────────────────
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

# ── Extra features extractor ────────────────────────────────
def extract_extra_features(texts):
    """
    Extracts word counts and phrase presence as extra features.
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

# ── Main preprocessing ───────────────────────────────────────
def main():
    csv_path = os.path.join(DATA_PROC, "combined_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("combined_dataset.csv not found. Run generate_dataset.py first.")

    print("Loading dataset...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"  Total records : {len(df)}")
    print(f"  Real (1)      : {(df['label'] == 1).sum()}")
    print(f"  Fake (0)      : {(df['label'] == 0).sum()}")

    df = df[['content', 'label']].copy()

    # Clean text
    print("\nCleaning text...")
    df['cleaned'] = df['content'].apply(clean_text)
    df = df[df['cleaned'].str.strip() != ""].reset_index(drop=True)
    print(f"  Records after cleaning: {len(df)}")

    X = df['cleaned'].values
    y = df['label'].values

    # Train / Val / Test split
    print("\nSplitting dataset (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")
    print(f"  Test  : {len(X_test)} samples")

    # ── TF-IDF Vectorization (1-3 grams) ───────────────────────
    print("\nFitting TF-IDF vectorizer (max_features=5000, ngram_range=(1,3))...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"  Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix  : {X_train_tfidf.shape}")

    # Add extra features (Boosted by repeating columns 2x)
    print("Adding extra features (word counts + suspicious phrases)...")
    extra_train = extract_extra_features(X_train)
    # Boost: Repeat extra features 2 times instead of 10 to avoid overpowering TF-IDF
    extra_train_boosted = np.tile(extra_train, 2)
    
    X_train_extra = hstack([X_train_tfidf, extra_train_boosted])
    X_val_extra   = hstack([X_val_tfidf, np.tile(extract_extra_features(X_val), 2)])
    X_test_extra  = hstack([X_test_tfidf, np.tile(extract_extra_features(X_test), 2)])
    print(f"  Feature matrix after boosting extra features: {X_train_extra.shape}")

    # Save TF-IDF vectorizer
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nVectorizer saved -> {vec_path}")

    # Save processed splits
    splits = {
        "X_train": X_train_extra,
        "X_val":   X_val_extra,
        "X_test":  X_test_extra,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
    }
    for name, data in splits.items():
        path = os.path.join(DATA_PROC, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
    print("Data splits saved to data/processed/")
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()

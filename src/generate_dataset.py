import pandas as pd
import os
import re

# TEXT CLEANING FUNCTION
def clean_text(text):
    text = str(text)

    # Remove newlines, tabs
    text = re.sub(r'[\n\r\t]', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters (keep letters and punctuation)
    text = re.sub(r'[^a-zA-Z.,!? ]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Lowercase
    text = text.lower()

    return text.strip()


# LOAD WELFAKE DATASET

def load_welfake(path, sample_size=5000):
    print("Loading WELFake dataset...")
    df = pd.read_csv(path)

    # Keep required columns
    df = df[['title', 'text', 'label']]

    # Remove missing values
    df = df.dropna()

    # Sample for balance
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)

    # Add source column
    df['source'] = 'welfake'

    # Handle label flip: WELFake originally uses 1=Fake, 0=Real.
    # We want 1=Real, 0=Fake consistently across all datasets.
    df['label'] = 1 - df['label']

    return df

# LOAD NEPAL DATASET
def load_nepal_dataset(path, label_value, source_name):
    print(f"Loading {source_name} dataset...")
    df = pd.read_csv(path)

    df = df[['title', 'text']]
    df = df.dropna()

    df['label'] = label_value
    df['source'] = source_name

    return df


# MAIN FUNCTION
def main():

    # File paths
    welfake_path = "data/raw/WELFake_Dataset.csv"
    nepal_real_path = "data/raw/nepal_real_news.csv"
    nepal_fake_path = "data/raw/nepal_fake_news.csv"

    # Load datasets
    welfake = load_welfake(welfake_path, sample_size=5000)
    nepal_real = load_nepal_dataset(nepal_real_path, 1, "nepal_real")
    nepal_fake = load_nepal_dataset(nepal_fake_path, 0, "nepal_fake")

    print("Combining datasets...")

    # Combine
    combined = pd.concat([welfake, nepal_real, nepal_fake], ignore_index=True)

    # Create content column
    combined['content'] = combined['title'] + " " + combined['text']

    print("Cleaning text...")
    combined['content'] = combined['content'].apply(clean_text)

    # Shuffle dataset
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Show dataset info
    print("\nFinal Dataset Shape:", combined.shape)
    print("\nLabel Distribution:")
    print(combined['label'].value_counts())

    # Create processed folder if not exists
    os.makedirs("data/processed", exist_ok=True)

    # Save
    output_path = "data/processed/combined_dataset.csv"
    combined.to_csv(output_path, index=False)

    print("\nDataset saved successfully to:", output_path)


if __name__ == "__main__":
    main()

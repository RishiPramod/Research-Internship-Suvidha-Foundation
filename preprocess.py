import os
import re
import string
import pickle
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

from config import VOCAB_PATH
from train_lstm import TRAIN_CSV

RAW_PATH = r"C:\Users\saisa\OneDrive\Desktop\Rishi\The_Hindu_paper_dataset_2015_to_2025.csv"
OUT_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data"

VOCAB_PATH = os.path.join(OUT_DIR, "vocab.pkl")
TRAIN_CSV = os.path.join(OUT_DIR, "train.csv")
TEST_CSV = os.path.join(OUT_DIR, "test.csv")

os.makedirs(OUT_DIR, exist_ok=True)

BOILERPLATE_REGEX = re.compile(r"\b(e-paper|e paper|the view from india|view from india)\b", flags=re.I)
DATE_REGEX = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4}\b", flags=re.I)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t=text
    t=BOILERPLATE_REGEX.sub("", t)
    t=DATE_REGEX.sub("", t)
    t=re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t)

    t=t.strip().lower()
    t=t.translate(PUNCT_TABLE)
    t=re.sub(r"\d+", " ", t)
    t=re.sub(r"\s+", " ", t)
    return t

def build_vocab(texts, min_freq=2,max_size=40000):
    cnt=Counter()
    for t in texts:
        cnt.update(t.split())

    items=[tok for tok, f in cnt.most_common(max_size) if f >= min_freq]
    vocab={tok: idx+2 for idx,tok in enumerate(items)}
    vocab.update({"pad":0, "unk":1})
    return vocab

def main():
    print("Loading data...")
    df=pd.read_csv(RAW_PATH)
    print("Cleaning content and headline...")
    df["content_cleaned"] = df["Content"].astype(str).apply(clean_text)
    df["headline_cleaned"] = df["Headline"].astype(str).apply(clean_text)

    def extract_section(url):
        try:
            parts = url.split('/')
            for p in parts:
                if p in ("sports", "business", "entertainment", "technology", "health"):
                    return p
        except Exception as e:
            return "other"
        return "other"
    
    print("Extracting sections from URL (if possible)...")
    df["section"] = df["URL"].astype(str).apply(extract_section)

    # Drop rows with empty content after cleaning
    df = df[df["content_cleaned"].str.strip() != ""].reset_index(drop=True)

    print(f"Total cleaned articles: {len(df)}")

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Build vocab from train content
    print("Building vocab from training data...")
    vocab = build_vocab(train_df["content_cleaned"].tolist(), min_freq=2, max_size=60000)
    print(f"Vocab size (incl. PAD/UNK): {len(vocab)}")

    # Save artifacts
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("Saved:")
    print(" - ", TRAIN_CSV)
    print(" - ", TEST_CSV)
    print(" - ", VOCAB_PATH)

if __name__ == "__main__":
    main() 
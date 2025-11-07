def decode_text(ids, vocab):
    # Reverse vocab: id -> token
    id2token = {v: k for k, v in vocab.items()}
    tokens = [id2token.get(int(i), "unk") for i in ids if int(i) != vocab.get("pad", 0)]
    return " ".join(tokens)
import os
import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import VOCAB_PATH
from rouge import Rouge

VOCAB_PATH = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\vocab.pkl"

def load_vocab(path=VOCAB_PATH):
    with open(path, "rb") as f:
        vocab=pickle.load(f)
    return vocab

def encode_text(text: str, vocab: dict, max_len: int=256) -> List[int]:
    if not isinstance(text, str) or text.strip() == "":
        return [0]*max_len

    toks=text.split()[:max_len]
    ids=[int(vocab.get(t, vocab.get("unk", 0)) or 0) for t in toks]
    if len(ids)<max_len:
        ids+=[int(vocab.get("pad", 0))]*(max_len-len(ids))
    return ids

def encode_texts(texts: List[str], vocab: dict, max_len: int=256) -> np.ndarray:
    return np.array([encode_text(t,vocab,max_len) for t in texts],dtype=np.int64)

def compute_classification_scores(y_true: List[str], y_pred: List[str]) -> dict:
# y_true / y_pred are text labels (strings). For deep models we will map predictions to nearest headline token (see model scripts).
# For stability, handle empty lists
    if len(y_true) == 0:
        return {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}


    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def compute_rouge_scores(hyps: List[str], refs: List[str]) -> dict:
    rouge = Rouge()
    # Rouge expects non-empty strings; replace empty with single token
    hyps = [h if isinstance(h, str) and h.strip() != "" else "empty" for h in hyps]
    refs = [r if isinstance(r, str) and r.strip() != "" else "empty" for r in refs]
    try:
        scores = rouge.get_scores(hyps, refs, avg=True)
        if isinstance(scores, list):
            scores = scores[0]
        return {
            "ROUGE-1": scores["rouge-1"]["f"],
            "ROUGE-2": scores["rouge-2"]["f"],
            "ROUGE-L": scores["rouge-l"]["f"]
        }
    except Exception as e:
        print("ROUGE calculation failed:", e)
        return {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}

def save_metrics_table(metrics: dict, out_path: str = "./results/metrics_summary.csv"):
# metrics: {model_name: {metric_name: value, ...}, ...}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for model, vals in metrics.items():
        row = {"model": model}
        row.update(vals)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved metrics summary to {out_path}")
    
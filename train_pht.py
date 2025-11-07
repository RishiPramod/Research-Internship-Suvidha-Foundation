import os
import time
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Rouge metric
from rouge import Rouge

# utils from preprocess step
from train_cnn import TRAIN_CSV
from utils import load_vocab, encode_texts, compute_rouge_scores

TRAIN_CSV = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\train.csv"
TEST_CSV = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\test.csv"
VOCAB_PATH = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\vocab.pkl"
MODEL_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\models"
RESULTS_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EMBED_DIM = 300  # GloVe embedding size
TRANSFORMER_LAYERS = 6
NHEAD = 6
ENC_FF_DIM = 768
DEC_HIDDEN = 384
DEC_NUM_LAYERS = 3
BATCH_SIZE = 32
LR = 1e-4  # Lower learning rate for AdamW
EPOCHS = 10
MAX_SRC_LEN = 300
MAX_TGT_LEN = 15  # Reduce target length for headlines
CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# -------------------------
# Utilities: load vocab / encode
# -------------------------
def load_vocab(path=VOCAB_PATH):
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    return vocab

def load_glove_embeddings(vocab, glove_path="glove.6B.300d.txt"):
    embeddings = np.random.normal(scale=0.6, size=(len(vocab), EMBED_DIM))
    found = 0
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            idx = vocab.get(word)
            if idx is not None:
                embeddings[idx] = vec
                found += 1
    print(f"Loaded {found} GloVe vectors.")
    return torch.tensor(embeddings, dtype=torch.float)

def encode_texts(texts, vocab, max_len):
    pad = vocab.get("<pad>", 0)
    unk = vocab.get("<unk>", 1)
    arr = []
    for t in texts:
        if not isinstance(t, str):
            t = ""
        t = t.strip().lower()
        t = " ".join([tok for tok in t.split() if len(tok) > 2])  # Remove very short tokens
        toks = t.split()[:max_len]
        ids = [vocab.get(tok, unk) for tok in toks]
        if len(ids) < max_len:
            ids = ids + [pad] * (max_len - len(ids))
        arr.append(ids)
    return np.array(arr, dtype=np.int64)

# -------------------------
# Dataset
# -------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN):
        self.src = encode_texts(df['content_cleaned'].astype(str).tolist(), vocab, max_len=max_src)
        pad = vocab.get("<pad>", 0)
        unk = vocab.get("<unk>", 1)
        tgt_list = []
        for h in df['headline_cleaned'].astype(str).tolist():
            toks = h.split()[:max_tgt]
            ids = [vocab.get(t, unk) for t in toks]
            if len(ids) < max_tgt:
                ids = ids + [pad] * (max_tgt - len(ids))
            tgt_list.append(ids)
        self.tgt = np.array(tgt_list, dtype=np.int64)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx], dtype=torch.long), torch.tensor(self.tgt[idx], dtype=torch.long)

# -------------------------
# Positional encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pos_emb", pe)
        self.pos_emb: torch.Tensor  # 👈 helps Pylance understand

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_emb[:, :x.size(1), :].to(x.device)

# -------------------------
# Models: Encoder (Transformer) + Decoder (LSTM)
# -------------------------
class PHTEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, nhead, ff_dim, dropout=0.3, max_len=MAX_SRC_LEN, glove_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if glove_weights is not None:
            self.embedding.weight.data.copy_(glove_weights)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, src, src_key_padding_mask=None):
        emb = self.embedding(src) * np.sqrt(EMBED_DIM)
        emb = self.pos_enc(emb)
        emb = self.ln(emb)
        out = self.transformer(emb.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2)
        out = self.ln(out)
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.5, glove_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if glove_weights is not None:
            self.embedding.weight.data.copy_(glove_weights)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)  # bidirectional
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def init_hidden(self, enc_outputs):
        mean = enc_outputs.mean(dim=1)
        h0 = torch.tanh(self.init_h(mean)).unsqueeze(0).repeat(DEC_NUM_LAYERS * 2, 1, 1)  # bidirectional
        c0 = torch.tanh(self.init_c(mean)).unsqueeze(0).repeat(DEC_NUM_LAYERS * 2, 1, 1)
        return (h0, c0)

    def forward_step(self, input_step, hidden):
        emb = self.embedding(input_step).unsqueeze(1)
        emb = self.ln(emb)
        output, hidden = self.lstm(emb, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden

class PHTSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # src: [B,S], tgt: [B,T]
        batch_size = src.size(0)
        tgt_len = tgt.size(1) if tgt is not None else MAX_TGT_LEN
        vocab_size = self.decoder.out.out_features

        # mask where src is pad
        src_key_padding_mask = (src == 0)  # True at padding positions
        enc_out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # [B,S,E]

        hidden = self.decoder.init_hidden(enc_out)
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        input_step = torch.full((batch_size,), fill_value=0, dtype=torch.long, device=src.device)  # start token = PAD (0)

        for t in range(tgt_len):
            logits, hidden = self.decoder.forward_step(input_step, hidden)
            outputs[:, t, :] = logits
            if (tgt is not None) and (np.random.rand() < teacher_forcing_ratio):
                input_step = tgt[:, t]
            else:
                input_step = logits.argmax(dim=1)
        return outputs

# -------------------------
# Metrics
# -------------------------
def token_overlap_metrics(hyps, refs):
    precisions, recalls, f1s = [], [], []
    exact = 0
    for h, r in zip(hyps, refs):
        if h.strip() == r.strip():
            exact += 1
        h_tokens = h.split()
        r_tokens = r.split()
        if len(h_tokens) == 0 or len(r_tokens) == 0:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0); continue
        overlap = len(set(h_tokens) & set(r_tokens))
        p = overlap / len(h_tokens)
        rr = overlap / len(r_tokens)
        f = 0.0 if (p + rr) == 0 else 2*p*rr / (p + rr)
        precisions.append(p); recalls.append(rr); f1s.append(f)
    return {
        'Accuracy_exact_match': exact / len(hyps),
        'Precision_token': float(np.mean(precisions)),
        'Recall_token': float(np.mean(recalls)),
        'F1_token': float(np.mean(f1s))
    }

def compute_rouge_scores(hyps, refs):
    rouge = Rouge()
    # clean empty
    hyps = [h if isinstance(h, str) and h.strip() != "" else "empty" for h in hyps]
    refs = [r if isinstance(r, str) and r.strip() != "" else "empty" for r in refs]
    try:
        scores = rouge.get_scores(hyps, refs, avg=True)
        return {
            "ROUGE-1": scores["rouge-1"]["f"] if isinstance(scores, dict) else scores[0]["rouge-1"]["f"],
            "ROUGE-2": scores["rouge-2"]["f"] if isinstance(scores, dict) else scores[0]["rouge-2"]["f"],
            "ROUGE-L": scores["rouge-l"]["f"] if isinstance(scores, dict) else scores[0]["rouge-l"]["f"]
        }
    except Exception as e:
        print("Rouge failed:", e)
        return {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}

# -------------------------
# Training & Evaluation
# -------------------------
def train():
    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    inv_vocab = {v:k for k,v in vocab.items()}

    df = pd.read_csv(TRAIN_CSV)
    # Remove duplicates and very short samples
    df = df.drop_duplicates(subset=["content_cleaned", "headline_cleaned"])
    df = df[df["content_cleaned"].str.len() > 20]
    df = df[df["headline_cleaned"].str.len() > 5]
    # Validation split
    val_frac = 0.1
    val_size = int(len(df) * val_frac)
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]
    test_df = pd.read_csv(TEST_CSV)
    train_ds = Seq2SeqDataset(train_df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN)
    val_ds = Seq2SeqDataset(val_df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN)
    test_ds = Seq2SeqDataset(test_df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    glove_path = "glove.6B.300d.txt"  # Download and place in your workspace
    glove_weights = load_glove_embeddings(vocab, glove_path)
    encoder = PHTEncoder(vocab_size, EMBED_DIM, TRANSFORMER_LAYERS, NHEAD, ENC_FF_DIM, dropout=0.3, max_len=MAX_SRC_LEN, glove_weights=glove_weights).to(DEVICE)
    decoder = LSTMDecoder(vocab_size, EMBED_DIM, DEC_HIDDEN, num_layers=DEC_NUM_LAYERS, dropout=0.5, glove_weights=glove_weights).to(DEVICE)
    model = PHTSeq2Seq(encoder, decoder).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get("<pad>", 0))

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        start = time.time()
        for src_batch, tgt_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=0.7)
            B, T, V = outputs.size()
            loss = criterion(outputs.contiguous().view(B*T, V), tgt_batch.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()
        elapsed = time.time() - start
        avg_train_loss = total_loss / len(train_loader)
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(DEVICE)
                tgt_batch = tgt_batch.to(DEVICE)
                outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=0.0)
                B, T, V = outputs.size()
                loss = criterion(outputs.contiguous().view(B*T, V), tgt_batch.contiguous().view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | time {elapsed:.1f}s")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(f"{MODEL_DIR}", "pht_model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation (greedy)
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(test_loader, desc="Evaluating"):
            src_batch = src_batch.to(DEVICE)
            outputs = model(src_batch, tgt=None, teacher_forcing_ratio=0.0)  # greedy
            preds = outputs.argmax(dim=-1).cpu().numpy()  # [B, T]
            tgt_np = tgt_batch.numpy()
            for p_seq, t_seq in zip(preds, tgt_np):
                p_tokens = [inv_vocab.get(int(tok), "<unk>") for tok in p_seq]
                t_tokens = [inv_vocab.get(int(tok), "<unk>") for tok in t_seq]
                hyp = " ".join([tok for tok in p_tokens if tok not in ("<pad>",)])
                ref = " ".join([tok for tok in t_tokens if tok not in ("<pad>",)])
                hyps.append(hyp)
                refs.append(ref)

    overlap = token_overlap_metrics(hyps, refs)
    rouge = compute_rouge_scores(hyps, refs)

    metrics = {
        "Accuracy": overlap["Accuracy_exact_match"],
        "Precision": overlap["Precision_token"],
        "Recall": overlap["Recall_token"],
        "F1": overlap["F1_token"],
        "ROUGE-1": rouge["ROUGE-1"],
        "ROUGE-2": rouge["ROUGE-2"],
        "ROUGE-L": rouge["ROUGE-L"]
    }

    # save outputs
    pd.DataFrame({"generated": hyps, "reference": refs}).to_csv(os.path.join(RESULTS_DIR, "pht_predictions.csv"), index=False)
    pd.DataFrame([{"model": "PHT", **metrics}]).to_csv(os.path.join(RESULTS_DIR, "pht_metrics.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(f"{MODEL_DIR}", "pht_model.pt"))

    print("PHT training complete. Metrics:")
    print(metrics)

if __name__ == "__main__":
    train()
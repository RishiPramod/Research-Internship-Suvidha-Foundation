import os
import time  
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import load_vocab, encode_texts,compute_rouge_scores

from config import TRAIN_CSV
TEST_CSV = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\test.csv"
VOCAB_PATH = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\vocab.pkl"
MODEL_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\models"
RESULTS_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EMBED_DIM=300
ENC_HIDDEN=512
DEC_HIDDEN=512
NUM_LAYERS=2
BATCH_SIZE=32
NUM_EPOCHS=10
LEARNING_RATE=0.001
MAX_SRC_LEN=400
MAX_TRG_LEN=20
MAX_TGT_LEN=20
CLIP = 1.0  # Gradient clipping value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class Seq2SeqDataset(Dataset):
    def __init__(self, df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN):
        self.src = encode_texts(df['content_cleaned'].tolist(), vocab, max_len=max_src)
        # for targets, we will right-pad/truncate to max_tgt
        self.tgt = []
        for h in df['headline_cleaned'].astype(str).tolist():
            toks = h.split()[:max_tgt]
            ids = [vocab.get(t, vocab.get('<unk>',1)) for t in toks]
            if len(ids) < max_tgt:
                ids = ids + [vocab.get('<pad>',0)] * (max_tgt - len(ids))
            self.tgt.append(ids)
        self.tgt = np.array(self.tgt, dtype=np.int64)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx], dtype=torch.long), torch.tensor(self.tgt[idx], dtype=torch.long)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, src):
        # src: [B, S]
        emb = self.embedding(src) # [B, S, E]
        outputs, (h, c) = self.lstm(emb) # outputs [B, S, 2H]
        # combine bidirectional hidden states
        # h: [num_layers*2, B, H]
        return outputs, h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden, dec_hidden, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # project encoder hidden to decoder hidden
        self.bridge = nn.Linear(enc_hidden*2, dec_hidden)
        self.lstm = nn.LSTM(embed_dim, dec_hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(dec_hidden, vocab_size)

    def forward(self, input_step, hidden):
        # input_step: [B, 1]
        emb = self.embedding(input_step)
        output, hidden = self.lstm(emb, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden, dec_hidden, n_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, enc_hidden, n_layers=n_layers, dropout=dropout)
        self.decoder = Decoder(vocab_size, embed_dim, enc_hidden, dec_hidden, n_layers=n_layers, dropout=dropout)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # src: [B, S], tgt: [B, T]
        batch_size = src.size(0)
        vocab_size = self.decoder.out.out_features
        tgt_len = MAX_TGT_LEN


        enc_outputs, h, c = self.encoder(src)
        num_layers = h.size(0) // 2
        dec_h = []
        dec_c = []
        for i in range(num_layers):
            forward_h = h[2*i]
            backward_h = h[2*i+1]
            dec_h.append(torch.tanh(self.decoder.bridge(torch.cat([forward_h, backward_h], dim=1))))
            try:
                forward_c = c[2*i]
                backward_c = c[2*i+1]
                dec_c.append(torch.tanh(self.decoder.bridge(torch.cat([forward_c, backward_c], dim=1))))
            except Exception:
                dec_c.append(torch.zeros_like(dec_h[-1]))
        dec_h = torch.stack(dec_h, dim=0)
        dec_c = torch.stack(dec_c, dim=0)

        decoder_hidden = (dec_h, dec_c)

        input_step = torch.full((batch_size, 1), fill_value=0, dtype=torch.long, device=src.device)
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        for t in range(tgt_len):
            logits, decoder_hidden = self.decoder(input_step, decoder_hidden) # logits [B, V]
            outputs[:, t, :] = logits
            teacher_force = False
            if tgt is not None:
                teacher_force = np.random.rand() < teacher_forcing_ratio
            top1 = logits.argmax(1).unsqueeze(1)
            input_step = (tgt[:, t].unsqueeze(1) if (teacher_force and tgt is not None) else top1)

        return outputs

def token_overlap_metrics(hyps, refs):
# hyps, refs are lists of strings (cleaned headlines)
    precisions, recalls, f1s = [], [], []
    exact = 0
    for h, r in zip(hyps, refs):
        if h.strip() == r.strip():
            exact += 1
        h_tokens = h.split()
        r_tokens = r.split()
        if len(h_tokens) == 0 or len(r_tokens) == 0:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
            continue
        overlap = len(set(h_tokens) & set(r_tokens))
        p = overlap / len(h_tokens)
        r_ = overlap / len(r_tokens)
        if p + r_ == 0:
            f = 0.0
        else:
            f = 2 * p * r_ / (p + r_)
        precisions.append(p); recalls.append(r_); f1s.append(f)
    return {
        'Accuracy_exact_match': exact / len(hyps),
        'Precision_token': float(np.mean(precisions)),
        'Recall_token': float(np.mean(recalls)),
        'F1_token': float(np.mean(f1s))
    }

def train():
    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_ds = Seq2SeqDataset(train_df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN)
    test_ds = Seq2SeqDataset(test_df, vocab, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = Seq2Seq(vocab_size, EMBED_DIM, ENC_HIDDEN, DEC_HIDDEN, n_layers=NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get('<pad>',0))
    for epoch in range(1,NUM_EPOCHS+1):
        model.train()
        epoch_loss = 0
        start = time.time()
        for src_batch, tgt_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5) # [B, T, V]
            B, T, V = outputs.size()
            loss = criterion(outputs.view(B*T, V), tgt_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss: {epoch_loss/len(train_loader):.4f} time: {time.time()-start:.1f}s")

# --- evaluation ---
    model.eval()
    hyps = []
    refs = []
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(test_loader, desc="Evaluating"):
            src_batch = src_batch.to(device)
            outputs = model(src_batch, tgt=None, teacher_forcing_ratio=0.0) # greedy decode
            preds = outputs.argmax(-1).cpu().numpy() # [B, T]
            for p, t in zip(preds, tgt_batch.numpy()):
                # decode tokens to strings using inverse vocab
                inv_vocab = {v:k for k,v in vocab.items()}
                p_tokens = [inv_vocab.get(int(tok), '<unk>') for tok in p]
                t_tokens = [inv_vocab.get(int(tok), '<unk>') for tok in t]
                hyp = ' '.join([tok for tok in p_tokens if tok not in ('<pad>',)])
                ref = ' '.join([tok for tok in t_tokens if tok not in ('<pad>',)])
                hyps.append(hyp)
                refs.append(ref)

# token overlap metrics (used as Accuracy/Precision/Recall/F1 proxies)
    overlap = token_overlap_metrics(hyps, refs)
    rouge = compute_rouge_scores(hyps, refs)

    metrics = {
        'Accuracy': overlap['Accuracy_exact_match'],
        'Precision': overlap['Precision_token'],
        'Recall': overlap['Recall_token'],
        'F1': overlap['F1_token'],
        'ROUGE-1': rouge['ROUGE-1'],
        'ROUGE-2': rouge['ROUGE-2'],
        'ROUGE-L': rouge['ROUGE-L']
    }

# save predictions
    preds_df = pd.DataFrame({'generated': hyps, 'reference': refs})
    preds_df.to_csv(os.path.join(RESULTS_DIR, 'lstm_predictions.csv'), index=False)

# save metrics
    metrics_df = pd.DataFrame([{'model': 'LSTM', **metrics}])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'lstm_metrics.csv'), index=False)

# save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_model.pt'))

    print("LSTM training and evaluation complete. Metrics:")
    print(metrics)

if __name__ == "__main__":
    train()
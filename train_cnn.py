import os
import pandas as pd
import numpy as np
from rouge import Rouge
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from utils import load_vocab, encode_text, compute_classification_scores, compute_rouge_scores, save_metrics_table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN_CSV = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\train.csv"
TEST_CSV = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\test.csv"
VOCAB_PATH = r"C:\Users\saisa\OneDrive\Desktop\Rishi\data\vocab.pkl"
MODEL_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\models"
RESULTS_DIR = r"C:\Users\saisa\OneDrive\Desktop\Rishi\results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class Vocab:
    def __init__(self, tokens, min_freq=2):
        counter = Counter(tokens)
        self.token2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    def __len__(self):
        return len(self.token2idx)

    def encode(self, text):
        return [self.token2idx.get(t, self.token2idx["<unk>"]) for t in text.split()]

    def decode(self, ids):
        tokens = [self.idx2token.get(i, "<unk>") for i in ids]
        return " ".join([t for t in tokens if t not in ["<sos>", "<eos>", "<pad>"]])

class NewsDataset(Dataset):
    def __init__(self, df, vocab, max_len=50):
        from utils import encode_texts
        self.inputs = encode_texts(df["content_cleaned"].tolist(), vocab, max_len)
        self.targets = encode_texts(df["headline_cleaned"].tolist(), vocab, max_len)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        import numpy as np
        inp = np.concatenate(([1], self.inputs[idx], [2]))
        tgt = np.concatenate(([1], self.targets[idx], [2]))
        inp = np.pad(inp, (0, max(0, self.max_len - len(inp))), 'constant')
        tgt = np.pad(tgt, (0, max(0, self.max_len - len(tgt))), 'constant')
        return torch.tensor(inp), torch.tensor(tgt)

# ========================
# Model: CNN Encoder + LSTM Decoder
# ========================
class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, embed]
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch, num_filters * len(filter_sizes)]
        return cat

class Seq2SeqCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_filters, filter_sizes):
        super().__init__()
        self.encoder = CNNEncoder(vocab_size, embed_size, num_filters, filter_sizes)
        self.decoder = nn.LSTM(num_filters * len(filter_sizes), hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        outputs = torch.zeros(batch_size, trg_len, len(vocab)).to(src.device)

        encoder_out = self.encoder(src).unsqueeze(1)  # [batch, 1, feat_dim]
        hidden = None
        input_step = encoder_out

        for t in range(trg_len):
            print(f"input_step shape before decoder: {input_step.shape}")
            # Fix input_step shape if 4D
            if input_step.dim() == 4:
                input_step = input_step.squeeze(1)
            output, hidden = self.decoder(input_step, hidden)
            pred = self.fc(output.squeeze(1))
            outputs[:, t, :] = pred
            top1 = pred.argmax(1).unsqueeze(1)
            next_input = self.encoder.embedding(top1) if np.random.rand() < teacher_forcing_ratio else self.encoder.embedding(trg[:, t].unsqueeze(1))
            input_step = next_input.unsqueeze(1) if next_input.dim() == 2 else next_input
        return outputs

# ========================
# Training + Evaluation
# ========================
def train_model(model, train_loader, optimizer, criterion, vocab, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, vocab):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            pred_ids = output.argmax(2)
            from utils import decode_text
            for i in range(len(trg)):
                preds.append(decode_text(pred_ids[i].cpu().numpy(), vocab))
                targets.append(decode_text(trg[i].cpu().numpy(), vocab))

    # Extract categories from headlines
    def extract_category(text):
        for cat in ["sports", "business", "entertainment", "technology", "health"]:
            if cat in text.lower():
                return cat
        return "other"

    pred_cats = [extract_category(p) for p in preds]
    target_cats = [extract_category(t) for t in targets]

    metrics = {
        "Accuracy": accuracy_score(target_cats, pred_cats),
        "Precision": precision_score(target_cats, pred_cats, average="weighted", zero_division=0),
        "Recall": recall_score(target_cats, pred_cats, average="weighted", zero_division=0),
        "F1": f1_score(target_cats, pred_cats, average="weighted", zero_division=0)
    }

    rouge = Rouge()
    rouge_scores = rouge.get_scores(preds, targets, avg=True)
    if isinstance(rouge_scores, list):
        rouge_scores = rouge_scores[0]
    metrics.update({
        "ROUGE-1": rouge_scores.get("rouge-1", {}).get("f", 0.0),
        "ROUGE-2": rouge_scores.get("rouge-2", {}).get("f", 0.0),
        "ROUGE-L": rouge_scores.get("rouge-l", {}).get("f", 0.0)
    })

    pd.DataFrame({"Prediction": preds, "Target": targets}).to_csv(f"{RESULTS_DIR}/cnn_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(f"{RESULTS_DIR}/cnn_metrics.csv", index=False)

    return metrics

# ========================
# Main
# ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    train_dataset = NewsDataset(train_df, vocab)
    test_dataset = NewsDataset(test_df, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = Seq2SeqCNN(len(vocab), embed_size=300, hidden_size=256, num_filters=100, filter_sizes=[3,4,5]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train & Evaluate
    train_model(model, train_loader, optimizer, criterion, vocab, num_epochs=10)
    metrics = evaluate_model(model, test_loader, vocab)

    # Save model
    torch.save(model.state_dict(), f"{MODEL_DIR}/cnn_model.pt")
    print("CNN Training complete. Metrics:", metrics)
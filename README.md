# News Headline Generation Research

This repository contains experiments comparing multiple neural approaches (CNN-based, LSTM seq2seq, and PHT: Transformer encoder + LSTM decoder hybrid) for generating news headlines from article content.

## Project Structure

```
config.py                # Shared configuration constants
preprocess.py            # Data cleaning, vocab building
utils.py                 # Helper functions (encoding, decoding, ROUGE, etc.)
train_cnn.py             # CNN-based headline generation model
train_lstm.py            # LSTM seq2seq baseline
train_pht.py             # PHT hybrid model (Transformer encoder + LSTM decoder)
train_nb.py / train_svm.py / train_logistic.py  # Classical baselines (if used)
evaluate.py              # Central evaluation script (optional)
data/                    # Train/test CSVs (ignored in git)
models/                  # Saved model weights (.pt) (ignored in git)
results/                 # Metrics & predictions (ignored in git)
```

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Place pre-trained GloVe embeddings (`glove.6B.300d.txt`) in the repository root if using the PHT model with embeddings.

## Data Preparation

Run preprocessing to clean raw dataset, build vocabulary, and produce train/test splits:

```bash
python preprocess.py
```

## Training

Train individual models:

```bash
python train_cnn.py
python train_lstm.py
python train_pht.py
```

Each script saves metrics under `results/` and model weights under `models/`.

## Evaluation

Use ROUGE and token overlap metrics. For classification-style metrics (Accuracy, Precision, Recall, F1) headline matching is approximated via token overlap.
 
## Results

The following table summarizes the latest evaluation metrics (generated values saved under `results/`). Values were computed on the held-out test split using token-overlap and ROUGE metrics.

| Model | Accuracy | Precision | Recall | F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.9600 | 0.9216 | 0.9600 | 0.9404 | 0.3775 | 0.0624 | 0.3700 |
| LSTM | 0.0000 | 0.0288 | 0.0288 | 0.0288 | 0.1310 | 0.0261 | 0.1310 |
| PHT (this work) | 0.0000 | 0.0381 | 0.0381 | 0.0381 | 0.1309 | 0.0261 | 0.1309 |

Notes:
- The classification-style metrics (Accuracy/Precision/Recall/F1) are based on token-overlap / exact-match heuristics and may not fully capture generative quality; ROUGE scores measure n-gram overlap and are commonly used for headline generation evaluation.
- Exact numeric values are saved in `results/cnn_metrics.csv`, `results/lstm_metrics.csv`, and `results/pht_metrics.csv`.
- For reproducibility, run the preprocessing and training scripts listed above and check the `results/` folder for updated metrics after each experiment.

## Improving PHT Model

Key tuning strategies already implemented:
- Transformer depth & attention heads increased
- Bidirectional LSTM decoder
- Pre-trained GloVe embeddings (300d)
- LayerNorm + Dropout
- Validation split & early stopping

Potential further improvements:
- Scheduled sampling decay for teacher forcing
- Beam search decoding instead of greedy
- Coverage or pointer mechanisms for long source texts
- Better subword tokenization (e.g., SentencePiece or BPE)

## Reproducibility

Set random seeds if strict reproducibility is required:
```python
import torch, random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```


---
Feel free to open issues or pull requests for enhancements.

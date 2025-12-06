# Arabic Text Diacritization System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Natural Language Processing (NLP) pipeline for automatic Arabic diacritization. This project implements two state-of-the-art approaches: a feature-rich **BiLSTM-CRF** baseline and a fine-tuned **Transformer (AraBERT)** model. It is designed for high performance, modularity, and ease of deployment.

---

## 🚀 Key Features

### 1. Advanced BiLSTM-CRF Architecture
A robust sequence labeling model incorporating multiple linguistic features:
- **Character Embeddings**
- **Word Embeddings**
- **FastText Embeddings** (optional)
- **Bag-of-Words (BoW)** sentence features
- **TF-IDF** sentence features
- **CRF Layer** for modeling diacritic transitions

### 2. Transformer Fine-Tuning
- Fine-tunes **AraBERT** or similar models for token classification  
- Automatic subword alignment  
- Uses HuggingFace `Trainer` API  

### 3. Production Ready
- **Flask API** for real-time web demo  
- **Config-driven design**  
- **Clean inference pipeline**

---

## 📂 Project Structure

```text
├── build_vocab.py           # Build vocabularies & feature vectorizers
├── requirements.txt         # Python dependencies
├── setup.sh                 # Optional setup script
├── data/
│   ├── train.txt
│   ├── val.txt
│   └── fasttext_wiki.ar.vec # Optional FastText vectors
├── outputs/
│   ├── models/              # Model checkpoints
│   ├── logs/
│   └── processed/           # vocab + feature vectorizers
└── src/
    ├── app/                 # Flask application
    ├── config.py            # Hyperparameters & flags
    ├── features.py          # BoW, TF-IDF, FastText feature logic
    ├── preprocess.py        # Normalization & label extraction
    ├── data/                # Dataset + collate function
    ├── models/              # BiLSTM-CRF & Transformer models
    ├── train/               # Training scripts
    ├── infer/               # Inference scripts
    └── eval/                # DER evaluation
```

---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.8+
- (Optional but recommended) CUDA-compatible GPU

### 2. Setup

```bash
git clone https://github.com/kariem-magdy/NLP-project.git
cd "NLP project"
pip install -r requirements.txt
```

---

## 📊 Data Preparation

Before training, generate vocabularies and feature vectorizers.

1. Ensure `train.txt` and `val.txt` are inside the **data/** folder  
2. (Optional) Download FastText vectors and rename to:

```
data/fasttext_wiki.ar.vec
```

3. Run:

```bash
python build_vocab.py
```

Artifacts will be saved to:  
`outputs/processed/`

---

## 🧠 Usage

### 🔹 Train BiLSTM-CRF Model

```bash
python -m src.train.train_bilstm
```

Output:
- `outputs/models/best_bilstm.pt`
- DER logged to console/logs

---

### 🔹 Train Transformer (AraBERT)

```bash
python -m src.train.train_transformer
```

---

### 🔹 Inference (CLI)

```bash
python -m src.infer.infer
```

Modify the test sentence inside `src/infer/infer.py`.

---

### 🔹 Web Demo (Flask)

```bash
python -m src.app.app
```

Visit:

```
http://localhost:5000
```

---

## 📉 Evaluation Metric

The system uses **Diacritic Error Rate (DER)**:

```
DER = (Incorrect Predictions / Total Valid Characters) × 100%
```

---

## ⚙️ Configuration (`src/config.py`)

```python
# Feature Flags
use_word_emb = True
use_fasttext = False
use_bow = True
use_tfidf = True

# Hyperparameters
char_emb_dim = 128
lstm_hidden = 256
batch_size = 32
epochs = 20
lr = 1e-3
```

---

## 📜 License

This project is licensed under the **MIT License**.

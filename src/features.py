# src/features.py
import os
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import cfg

class FeatureManager:
    def __init__(self):
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        
    def fit(self, texts):
        print("[INFO] Fitting BoW and TF-IDF Vectorizers...")
        if cfg.use_bow:
            self.bow_vectorizer = CountVectorizer(max_features=cfg.bow_vocab_size, binary=False)
            self.bow_vectorizer.fit(texts)
        if cfg.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=cfg.tfidf_vocab_size)
            self.tfidf_vectorizer.fit(texts)
        self.save()

    def transform_bow(self, text):
        if not self.bow_vectorizer: return torch.zeros(0)
        try:
            vec = self.bow_vectorizer.transform([text]).toarray()[0]
            return torch.tensor(vec, dtype=torch.float)
        except: return torch.zeros(0)

    def transform_tfidf(self, text):
        if not self.tfidf_vectorizer: return torch.zeros(0)
        try:
            vec = self.tfidf_vectorizer.transform([text]).toarray()[0]
            return torch.tensor(vec, dtype=torch.float)
        except: return torch.zeros(0)

    def save(self):
        os.makedirs(cfg.processed_dir, exist_ok=True)
        if self.bow_vectorizer:
            with open(os.path.join(cfg.processed_dir, 'bow.pkl'), 'wb') as f: pickle.dump(self.bow_vectorizer, f)
        if self.tfidf_vectorizer:
            with open(os.path.join(cfg.processed_dir, 'tfidf.pkl'), 'wb') as f: pickle.dump(self.tfidf_vectorizer, f)

    def load(self):
        bow_path = os.path.join(cfg.processed_dir, 'bow.pkl')
        tfidf_path = os.path.join(cfg.processed_dir, 'tfidf.pkl')
        try:
            if cfg.use_bow and os.path.exists(bow_path):
                with open(bow_path, 'rb') as f: self.bow_vectorizer = pickle.load(f)
            if cfg.use_tfidf and os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f: self.tfidf_vectorizer = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load feature pickles: {e}")

    def load_fasttext_matrix(self, word2idx):
        if not os.path.exists(cfg.fasttext_path):
            print(f"[WARN] FastText file not found at {cfg.fasttext_path}. Skipping.")
            return None
        
        print("[INFO] Loading FastText...")
        vocab_size = len(word2idx)
        matrix = np.zeros((vocab_size, cfg.fasttext_dim))
        found = 0
        
        with open(cfg.fasttext_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0 and len(line.split()) < 10: continue
                parts = line.rstrip().split(' ')
                word = parts[0]
                if word in word2idx:
                    try:
                        matrix[word2idx[word]] = np.array(parts[1:], dtype=float)
                        found += 1
                    except: pass
        print(f"[INFO] FastText found {found} words.")
        return torch.tensor(matrix, dtype=torch.float)

feature_mgr = FeatureManager()
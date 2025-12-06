# src/infer/infer.py
import torch
import os
import sys
from ..config import cfg
from ..preprocess import load_json, extract_labels
from ..models.bilstm_crf import BiLSTMCRF
from ..features import feature_mgr

class DiacriticPredictor:
    def __init__(self):
        print(f"[INFO] Loading resources on {cfg.device}...")
        self.device = cfg.device
        
        # 1. Load Vocabs
        try:
            self.char2idx = load_json(os.path.join(cfg.processed_dir, "char2idx.json"))
            self.label2idx = load_json(os.path.join(cfg.processed_dir, "label2idx.json"))
            self.idx2label = {v: k for k, v in self.label2idx.items()}
        except FileNotFoundError:
            raise RuntimeError("Vocab files not found. Run build_vocab.py first.")

        # 2. Load Word Vocab (if needed)
        self.word2idx = None
        if cfg.use_word_emb or cfg.use_fasttext:
            try:
                self.word2idx = load_json(os.path.join(cfg.processed_dir, "word2idx.json"))
            except:
                print("[WARN] Word vocab not found. Feature disabled.")

        # 3. Load Feature Manager
        feature_mgr.load()

        # 4. Initialize Model
        # We pass None for fasttext_matrix because we load state_dict immediately after
        self.model = BiLSTMCRF(
            vocab_size=len(self.char2idx), 
            char_emb_dim=cfg.char_emb_dim, 
            lstm_hidden=cfg.lstm_hidden, 
            num_labels=len(self.label2idx), 
            word_vocab_size=len(self.word2idx) if self.word2idx else None,
            fasttext_matrix=None, 
            num_layers=cfg.lstm_layers, 
            dropout=0.0, # No dropout during inference
            pad_idx=0
        ).to(self.device)
        
        # 5. Load Weights
        ckpt_path = os.path.join(cfg.models_dir, 'best_bilstm.pt')
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found at {ckpt_path}")
            
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        # 6. CRITICAL: Set Eval Mode
        self.model.eval()
        print("[INFO] Model loaded and set to eval mode.")

    def predict(self, text):
        """
        Diacritizes a single input string.
        """
        self.model.eval() # Redundant safety check
        
        # 1. Preprocess
        text_clean, _ = extract_labels(text)
        if not text_clean: return ""

        # 2. Prepare Inputs
        chars = torch.tensor([self.char2idx.get(c, 1) for c in text_clean], dtype=torch.long).unsqueeze(0).to(self.device)
        
        word_ids = None
        if self.word2idx:
            w_ids = []
            words = text_clean.split()
            ptr = 0
            for c in text_clean:
                if c == ' ': 
                    w_ids.append(0)
                    ptr += 1
                elif ptr < len(words):
                    w_ids.append(self.word2idx.get(words[ptr], 1))
                else:
                    w_ids.append(0)
            word_ids = torch.tensor(w_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        bow = feature_mgr.transform_bow(text_clean).unsqueeze(0).to(self.device) if cfg.use_bow else None
        tfidf = feature_mgr.transform_tfidf(text_clean).unsqueeze(0).to(self.device) if cfg.use_tfidf else None
        
        # Mask (All 1s for inference)
        mask = torch.ones_like(chars, dtype=torch.bool)

        # 3. Forward
        with torch.no_grad():
            pred_ids = self.model(chars, word_ids, bow, tfidf, mask=mask)[0]

        # 4. Decode
        out = ""
        for char, pid in zip(text_clean, pred_ids):
            diacritic = self.idx2label.get(pid, '_')
            out += char + (diacritic if diacritic != '_' else '')
            
        return out

if __name__ == "__main__":
    pred = DiacriticPredictor()
    print(pred.predict("ذهب الطالب الى المدرسة"))
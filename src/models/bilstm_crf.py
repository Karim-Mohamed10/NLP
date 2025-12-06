# src/models/bilstm_crf.py
import torch
import torch.nn as nn
from .crf_layer import CRFLayer
from ..config import cfg

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, lstm_hidden, num_labels, 
                 word_vocab_size=None, fasttext_matrix=None, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        
        # 1. Chars
        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=pad_idx)
        input_dim = char_emb_dim
        
        # 2. Words (Trainable)
        if cfg.use_word_emb and word_vocab_size:
            self.word_emb = nn.Embedding(word_vocab_size, cfg.word_emb_dim, padding_idx=0)
            input_dim += cfg.word_emb_dim
            
        # 3. FastText (Frozen)
        self.ft_emb = None
        if cfg.use_fasttext and fasttext_matrix is not None:
            self.ft_emb = nn.Embedding.from_pretrained(fasttext_matrix, freeze=True, padding_idx=0)
            input_dim += cfg.fasttext_dim
            
        # 4. BoW Projector
        if cfg.use_bow:
            self.bow_proj = nn.Linear(cfg.bow_vocab_size, cfg.bow_emb_dim)
            input_dim += cfg.bow_emb_dim
            
        # 5. TF-IDF Projector
        if cfg.use_tfidf:
            self.tfidf_proj = nn.Linear(cfg.tfidf_vocab_size, cfg.tfidf_emb_dim)
            input_dim += cfg.tfidf_emb_dim

        self.bilstm = nn.LSTM(input_dim, lstm_hidden, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(2*lstm_hidden, num_labels)
        self.crf = CRFLayer(num_labels)

    def forward(self, chars, word_ids=None, bow=None, tfidf=None, mask=None, labels=None):
        # Base: Chars
        feats = [self.char_emb(chars)]
        
        # Add Words
        if hasattr(self, 'word_emb') and word_ids is not None:
            feats.append(self.word_emb(word_ids))
            
        if self.ft_emb is not None and word_ids is not None:
            feats.append(self.ft_emb(word_ids))
            
        # Expand Sentence Features
        seq_len = chars.size(1)
        if hasattr(self, 'bow_proj') and bow is not None:
            b = self.bow_proj(bow).unsqueeze(1).expand(-1, seq_len, -1)
            feats.append(b)
            
        if hasattr(self, 'tfidf_proj') and tfidf is not None:
            t = self.tfidf_proj(tfidf).unsqueeze(1).expand(-1, seq_len, -1)
            feats.append(t)
            
        # Concat & Pass
        x = torch.cat(feats, dim=2)
        out, _ = self.bilstm(x)
        emissions = self.linear(out)
        
        if labels is not None: return self.crf(emissions, labels, mask)
        return self.crf.decode(emissions, mask)
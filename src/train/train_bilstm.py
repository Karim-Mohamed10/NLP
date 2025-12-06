# src/train/train_bilstm.py
import os
import torch
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ..config import cfg
from ..preprocess import load_json
from ..data.dataset import DiacritizationDataset
from ..data.collate import collate_fn
from ..models.bilstm_crf import BiLSTMCRF
from ..utils.checkpoints import save_checkpoint
from ..features import feature_mgr
from ..eval.metrics import DERCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train")

def train():
    os.makedirs(cfg.models_dir, exist_ok=True)
    
    # 1. Load Vocabs
    try:
        char2idx = load_json(os.path.join(cfg.processed_dir, "char2idx.json"))
        label2idx = load_json(os.path.join(cfg.processed_dir, "label2idx.json"))
        # Create inverse mapping for evaluation
        idx2label = {v: k for k, v in label2idx.items()}
    except: 
        raise FileNotFoundError(f"Vocab files not found in {cfg.processed_dir}. Run build_vocab.py first.")
    
    # 2. Load Word Vocab & FastText (if enabled)
    word2idx, ft_matrix = None, None
    if cfg.use_word_emb or cfg.use_fasttext:
        try:
            word2idx = load_json(os.path.join(cfg.processed_dir, "word2idx.json"))
            if cfg.use_fasttext: 
                ft_matrix = feature_mgr.load_fasttext_matrix(word2idx)
        except: 
            logger.warning("word2idx.json missing. Word features will be disabled.")

    # 3. Create Datasets
    # Note: Using cfg.val_file as per your configuration
    train_ds = DiacritizationDataset(cfg.train_file, char2idx, label2idx, word2idx)
    val_ds = DiacritizationDataset(cfg.val_file, char2idx, label2idx, word2idx)
    
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, cfg.batch_size, collate_fn=collate_fn)

    # 4. Initialize Model
    model = BiLSTMCRF(
        vocab_size=len(char2idx), 
        char_emb_dim=cfg.char_emb_dim, 
        lstm_hidden=cfg.lstm_hidden, 
        num_labels=len(label2idx), 
        word_vocab_size=len(word2idx) if word2idx else None,
        fasttext_matrix=ft_matrix,
        num_layers=cfg.lstm_layers, 
        dropout=cfg.bilstm_dropout,
        pad_idx=0
    ).to(cfg.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_der = 100.0
    der_calc = DERCalculator()
    
    logger.info("Starting training...")
    
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        
        # Training Step
        for b in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move to device
            chars = b['chars'].to(cfg.device)
            labels = b['labels'].to(cfg.device)
            mask = b['mask'].to(cfg.device)
            
            # Optional features (check if they exist)
            word_ids = b['word_ids'].to(cfg.device) if b['word_ids'] is not None else None
            bow = b['bow'].to(cfg.device) if b['bow'] is not None else None
            tfidf = b['tfidf'].to(cfg.device) if b['tfidf'] is not None else None
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(chars, word_ids, bow, tfidf, mask, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Evaluation Step
        model.eval()
        
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            for b in val_loader:
                chars = b['chars'].to(cfg.device)
                mask = b['mask'].to(cfg.device)
                
                word_ids = b['word_ids'].to(cfg.device) if b['word_ids'] is not None else None
                bow = b['bow'].to(cfg.device) if b['bow'] is not None else None
                tfidf = b['tfidf'].to(cfg.device) if b['tfidf'] is not None else None
                
                # Predict (returns list of lists of indices)
                preds = model(chars, word_ids, bow, tfidf, mask=mask)
                
                # Process batch for DER calculation
                for i, p in enumerate(preds):
                    # Get valid length from mask to slice the prediction
                    # The CRF output is usually seq_len, so we must slice it by valid mask length
                    valid_len = int(mask[i].sum().item())
                    p_valid = p[:valid_len]
                    
                    # Map indices to labels
                    p_labels = [idx2label.get(x, '') for x in p_valid]
                    
                    # Get reference labels (these are already unpadded list of strings from collate)
                    r_labels = b['label_strs'][i]
                    
                    # Get raw text
                    raw = b['raws'][i]
                    
                    all_preds.append((raw, p_labels))
                    all_refs.append((raw, r_labels))

        # Metrics
        der = der_calc.compute(all_refs, all_preds) * 100
        avg_loss = total_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, DER={der:.2f}%")
        
        # Save Best Model
        if der < best_der:
            best_der = der
            save_checkpoint({
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'best_der': best_der,
                'char2idx': char2idx, 
                'label2idx': label2idx
            }, os.path.join(cfg.models_dir, 'best_bilstm.pt'))

if __name__ == "__main__":
    train()
# src/data/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(batch: List[Dict]):
    """
    Pads sequences and stacks features.
    """
    # Extract lists
    chars = [item["chars"] for item in batch]
    labels = [item["labels"] for item in batch]
    word_ids = [item["word_ids"] for item in batch]
    
    # Pad Sequences (0 is pad for all)
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=0)
    
    # Stack Fixed-Size Features
    # Check if they exist (numel > 0)
    if batch[0]["bow"].numel() > 0:
        bow = torch.stack([item["bow"] for item in batch])
    else:
        bow = None
        
    if batch[0]["tfidf"].numel() > 0:
        tfidf = torch.stack([item["tfidf"] for item in batch])
    else:
        tfidf = None

    # Create Mask
    mask = (chars_padded != 0)

    # Pass through raw data for metrics
    raws = [item["raw"] for item in batch]
    label_strs = [item["label_strs"] for item in batch]

    return {
        "chars": chars_padded,
        "labels": labels_padded,
        "word_ids": word_ids_padded,
        "bow": bow,
        "tfidf": tfidf,
        "mask": mask,
        "raws": raws,
        "label_strs": label_strs
    }
# src/preprocess.py
"""
Preprocessing module for Arabic diacritization.
Handles text cleaning, normalization, and label extraction (Shadda+Vowel aware).
"""

from typing import Tuple, List, Dict, Iterable, Optional, Any
import unicodedata
import regex as re
import json
import os
import logging
from collections import Counter, OrderedDict

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess")

# Constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Regex helpers
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF]") 
_COMBINING_RE = re.compile(r"\p{M}+")

def detect_encoding(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            _ = f.read(4096)
        return "utf-8"
    except UnicodeDecodeError as e:
        logger.error("File %s is not UTF-8 encoded: %s", path, e)
        raise

def clean_line(line: str, remove_urls=True, remove_emails=True, remove_zero_width=True, normalize_spaces=True) -> str:
    s = line
    if remove_urls: s = _URL_RE.sub(" ", s)
    if remove_emails: s = _EMAIL_RE.sub(" ", s)
    if remove_zero_width: s = _ZERO_WIDTH_RE.sub("", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    if normalize_spaces: s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_diacritics(line: str) -> str:
    return "".join(ch for ch in line if unicodedata.category(ch)[0] != "M")

def extract_labels(line: str) -> Tuple[str, List[str]]:
    """
    Robust extraction that handles:
    1. Combining marks (Shadda+Vowel) as single labels.
    2. Stray/Trailing diacritics.
    """
    base_chars = []
    labels = []
    pending_marks = []

    for ch in line:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            pending_marks.append(ch)
        else:
            if base_chars:
                # Attach pending marks to previous base char
                prev_label = labels[-1]
                # Combine with existing label if needed (e.g. if we had logical separation)
                # But typically pending_marks are the full set for the previous char here
                # We sort to ensure determinism (e.g. Shadda+Fatha == Fatha+Shadda)
                if pending_marks:
                     combined = "".join(sorted(pending_marks))
                     labels[-1] = combined if prev_label == "_" else prev_label + combined
                pending_marks = []
            
            base_chars.append(ch)
            labels.append("_") # Default empty label

    # Handle trailing marks (at end of sentence)
    if pending_marks and base_chars:
        combined = "".join(sorted(pending_marks))
        labels[-1] = combined if labels[-1] == "_" else labels[-1] + combined

    return "".join(base_chars), labels

def normalize_text(line: str, options: Optional[Dict[str, Any]] = None) -> str:
    if options is None: options = {}
    
    s = line
    s = unicodedata.normalize("NFKC", s)
    
    if options.get("remove_tatweel", True):
        s = s.replace("\u0640", "")
    
    if options.get("normalize_hamza", True):
        s = s.replace("\u0622", "\u0627").replace("\u0623", "\u0627").replace("\u0625", "\u0627")
    
    if options.get("lower_latin", True):
        s = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), s)

    if options.get("remove_punctuation", False):
        s = re.sub(r"[^\p{L}\p{N}\s]", " ", s)

    return re.sub(r"\s+", " ", s).strip()

def build_char_vocab(texts: Iterable[str], min_freq: int = 1):
    counter = Counter()
    for t in texts: counter.update(t)
    
    items = [(ch, freq) for ch, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], ord(x[0])))
    
    char2idx = OrderedDict()
    char2idx[PAD_TOKEN] = 0
    char2idx[UNK_TOKEN] = 1
    
    idx = 2
    for ch, _ in items:
        if ch not in char2idx:
            char2idx[ch] = idx
            idx += 1
            
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def build_label_map(labels: Iterable[List[str]]):
    s = set()
    for lablist in labels: s.update(lablist)
    if "_" not in s: s.add("_")
    
    # Sort deterministically
    sorted_labels = sorted([l for l in s if l != "_"], key=lambda x: (len(x), x))
    
    label2idx = OrderedDict({"_": 0})
    for i, lab in enumerate(sorted_labels, start=1):
        label2idx[lab] = i
        
    idx2label = {i: l for l, i in label2idx.items()}
    return label2idx, idx2label

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_file_to_entries(file_path: str, 
                          char2idx: Dict[str,int], 
                          label2idx: Dict[str,int],
                          strip_input_diacritics: bool = True, 
                          normalization_options: dict = None) -> List[Dict]:
    """
    Reads file and returns list of processed dictionaries.
    Used by the Dataset class.
    """
    entries = []
    detect_encoding(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = clean_line(line.rstrip("\n"))
            if not line: continue
            
            # 1. Extract Base + Labels using ROBUST logic
            base_text, labels = extract_labels(line)
            if not base_text: continue
            
            # 2. Normalize Base Text
            if normalization_options:
                base_text = normalize_text(base_text, normalization_options)
            
            # 3. Strip diacritics from input (redundant if normalization did it, but safe)
            if strip_input_diacritics:
                base_text = strip_diacritics(base_text)
                
            # 4. Map to IDs
            char_ids = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in base_text]
            
            # Safety align
            if len(labels) != len(base_text):
                labels = ["_"] * len(base_text) # Fallback
            
            label_ids = [label2idx.get(l, label2idx["_"]) for l in labels]
            
            entries.append({
                "raw": base_text,
                "labels": labels,
                "char_ids": char_ids,
                "label_ids": label_ids
            })
            
    return entries
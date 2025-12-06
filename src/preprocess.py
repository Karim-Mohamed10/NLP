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
    # Remove Control chars but keep everything else for now
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

def normalize_base_and_labels(base: str, labels: List[str], options: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """
    Normalizes the base text AND filters the labels list in sync.
    This prevents length mismatches when characters (like punctuation) are removed.
    """
    if not options:
        return base, labels

    out_chars = []
    out_labels = []

    rem_punct = options.get("remove_punctuation", False)
    rem_tatweel = options.get("remove_tatweel", True)
    norm_hamza = options.get("normalize_hamza", True)
    lower_lat = options.get("lower_latin", True)

    for char, label in zip(base, labels):
        # 1. Remove Tatweel
        if rem_tatweel and char == "\u0640":
            continue
            
        # 2. Lowercase Latin
        if lower_lat and 'A' <= char <= 'Z':
            char = char.lower()
            
        # 3. Normalize Hamza
        if norm_hamza:
            if char in "\u0622\u0623\u0625": char = "\u0627"
        
        # 4. Handle Punctuation / Non-Alphanumeric
        # If remove_punctuation is ON, we turn punct into SPACE (to separate words)
        # We do NOT delete it entirely here, or we risk merging words.
        # The space collapsing step later will handle excess spaces.
        if rem_punct:
            cat = unicodedata.category(char)
            # Keep Letters(L) and Numbers(N). Turn others to space.
            if not (cat.startswith('L') or cat.startswith('N')):
                char = " "
                label = "_" # Label for space is empty

        # 5. Stream Collapse Spaces
        if char.isspace():
            # If the last char we added was also a space, skip this one
            if out_chars and out_chars[-1] == " ":
                continue
            # Otherwise add a single space
            out_chars.append(" ")
            out_labels.append("_")
        else:
            out_chars.append(char)
            out_labels.append(label)

    # 6. Trim Leading/Trailing Spaces
    while out_chars and out_chars[0] == " ":
        out_chars.pop(0)
        out_labels.pop(0)
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_labels.pop()

    return "".join(out_chars), out_labels

def normalize_text(line: str, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Legacy function for text-only normalization (e.g. inference).
    """
    # Reuse the sync logic with dummy labels
    dummy_labels = ["_"] * len(line)
    norm_text, _ = normalize_base_and_labels(line, dummy_labels, options)
    return norm_text

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
            
            # 1. Extract Base + Labels
            base_text, labels = extract_labels(line)
            if not base_text: continue
            
            # 2. Normalize BOTH to keep them in sync
            # This fixes the bug where removing punct caused misalignment
            if normalization_options:
                base_text, labels = normalize_base_and_labels(base_text, labels, normalization_options)
            
            if not base_text: continue

            # 3. Map to IDs
            char_ids = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in base_text]
            label_ids = [label2idx.get(l, label2idx["_"]) for l in labels]
            
            # Safety Check (should pass now)
            if len(char_ids) != len(label_ids):
                # This should theoretically not happen with normalize_base_and_labels
                # But if it does, we must truncate to the shorter one to save data
                min_len = min(len(char_ids), len(label_ids))
                char_ids = char_ids[:min_len]
                label_ids = label_ids[:min_len]
                base_text = base_text[:min_len]
                labels = labels[:min_len]

            entries.append({
                "raw": base_text,
                "labels": labels,
                "char_ids": char_ids,
                "label_ids": label_ids
            })
            
    return entries
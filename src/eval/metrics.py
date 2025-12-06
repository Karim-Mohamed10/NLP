# src/eval/metrics.py
from typing import List, Tuple

class DERCalculator:
    """
    Compute Diacritic Error Rate (DER).
    DER = (Substitutions + Insertions + Deletions) / Total Reference Characters
    """

    def compute(self, refs: List[Tuple[str, List[str]]], preds: List[Tuple[str, List[str]]]):
        total_ref_chars = 0
        total_errs = 0
        
        for (r_raw, r_labels), (p_raw, p_labels) in zip(refs, preds):
            # 1. The Denominator is ALWAYS the Reference Length
            # (We trust refs are the correct, unpadded ground truth)
            total_ref_chars += len(r_labels)
            
            # 2. Compare character by character
            n = min(len(r_labels), len(p_labels))
            for i in range(n):
                if r_labels[i] != p_labels[i]:
                    total_errs += 1
            
            # 3. Penalize Length Mismatches (Insertions / Deletions)
            # If model predicts too few or too many items, those are errors.
            if len(r_labels) != len(p_labels):
                diff = abs(len(r_labels) - len(p_labels))
                total_errs += diff
                
        # Calculate Percentage
        der = total_errs / total_ref_chars if total_ref_chars > 0 else 0.0
        return der
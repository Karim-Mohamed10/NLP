# DER and simple utilities
from typing import List, Tuple

class DERCalculator:
    """
    Compute Diacritic Error Rate (DER).
    We define DER here as total wrong diacritics / total diacritic positions (characters).
    Inputs: lists of tuples [(raw_sentence, [pred_labels])], [(raw_sentence, [ref_labels])]
    """

    def compute(self, refs: List[Tuple[str, List[str]]], preds: List[Tuple[str, List[str]]]):
        assert len(refs) == len(preds)
        total = 0
        errs = 0
        for (r_raw, r_labels), (p_raw, p_labels) in zip(refs, preds):
            # assume same raw strings (base text)
            n = min(len(r_labels), len(p_labels))
            for i in range(n):
                total += 1
                if r_labels[i] != p_labels[i]:
                    errs += 1
            # if length mismatch, count remaining as errors
            if len(r_labels) != len(p_labels):
                diff = abs(len(r_labels)-len(p_labels))
                total += diff
                errs += diff
        der = errs / total if total>0 else 0.0
        return der

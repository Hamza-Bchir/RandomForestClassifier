import numpy as np
from collections import Counter

def compute_weighted_class_proportions(y, sample_weight):
    """Compute class proportions with sample weights. Returns dict {class_label: proportion}."""
    total = np.sum(sample_weight)
    if total == 0:
        # defensive: avoid division by zero
        return {c: 0.0 for c in np.unique(y)}
    proportions = {}
    for c in np.unique(y):
        mask = (y == c)
        proportions[c] = np.sum(sample_weight[mask]) / total
    return proportions


# impurity measures that accept either raw (y, sample_weight) or class-weight dict
class Gini:
    def __call__(self, y=None, sample_weight=None, class_weight_counts=None):
        if class_weight_counts is not None:
            # class_weight_counts: dict {class: weight_sum}
            total = sum(class_weight_counts.values())
            if total == 0:
                return 0.0
            return 1.0 - sum((w/total)**2 for w in class_weight_counts.values())
        proportions = compute_weighted_class_proportions(y, sample_weight)
        return 1.0 - sum(p**2 for p in proportions.values())


class MisclassificationError:
    def __call__(self, y=None, sample_weight=None, class_weight_counts=None):
        if class_weight_counts is not None:
            total = sum(class_weight_counts.values())
            if total == 0:
                return 0.0
            return 1.0 - max(w/total for w in class_weight_counts.values())
        proportions = compute_weighted_class_proportions(y, sample_weight)
        return 1.0 - max(proportions.values())


class Entropy:
    def __call__(self, y=None, sample_weight=None, class_weight_counts=None):
        if class_weight_counts is not None:
            total = sum(class_weight_counts.values())
            if total == 0:
                return 0.0
            s = 0.0
            for w in class_weight_counts.values():
                p = w / total
                if p > 0:
                    s -= p * np.log2(p)
            return s
        proportions = compute_weighted_class_proportions(y, sample_weight)
        s = 0.0
        for p in proportions.values():
            if p > 0:
                s -= p * np.log2(p)
        return s

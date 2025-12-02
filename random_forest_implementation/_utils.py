import numpy as np
from collections import Counter

def compute_weighted_class_proportions(y, sample_weight):
    """
    Compute class proportions using sample weights.

    Parameters
    ----------
    y : array-like
        Class labels.
    sample_weight : array-like
        Weight assigned to each sample.

    Returns
    -------
    dict
        Mapping {class_label: weighted_proportion}.
    """
    total = np.sum(sample_weight)
    if total == 0:
        return {c: 0.0 for c in np.unique(y)}

    proportions = {}
    for c in np.unique(y):
        mask = (y == c)
        proportions[c] = np.sum(sample_weight[mask]) / total
    return proportions


class Gini:
    """
    Gini impurity measure supporting both raw data (y, sample_weight)
    and pre-aggregated class-weight dictionaries.
    """
    def __call__(self, y=None, sample_weight=None, class_weight_counts=None):
        """
        Compute Gini impurity.

        Parameters
        ----------
        y : array-like, optional
            Class labels.
        sample_weight : array-like, optional
            Sample weights.
        class_weight_counts : dict, optional
            Precomputed {class_label: weight_sum}.

        Returns
        -------
        float
            Gini impurity.
        """
        if class_weight_counts is not None:
            total = sum(class_weight_counts.values())
            if total == 0:
                return 0.0
            return 1.0 - sum((w / total) ** 2 for w in class_weight_counts.values())

        proportions = compute_weighted_class_proportions(y, sample_weight)
        return 1.0 - sum(p ** 2 for p in proportions.values())


class MisclassificationError:
    """
    Misclassification error (1 - max class probability), supporting raw data
    and aggregated weight dictionaries.
    """
    def __call__(self, y=None, sample_weight=None, class_weight_counts=None):
        """
        Compute misclassification error.

        Parameters
        ----------
        y : array-like, optional
            Class labels.
        sample_weight : array-like, optional
            Sample weights.
        class_weight_counts : dict, optional
            Precomputed {class_label: weight_sum}.

        Returns
        -------
        float
            Misclassification impurity.
        """
        if class_weight_counts is not None:
            total = sum(class_weight_counts.values())
            if total == 0:
                return 0.0
            return 1.0 - max(w / total for w in class_weight_counts.values())

        proportions = compute_weighted_class_proportions(y, sample_weight)
        return 1.0 - max(proportions.values())


class Entropy:
    """
    Entropy impurity measure (Shannon entropy), supporting raw data
    and aggregated weight dictionaries.
    """
    def __call__(self, y=None, sample_weight=None, class_weight_counts=None):
        """
        Compute entropy impurity.

        Parameters
        ----------
        y : array-like, optional
            Class labels.
        sample_weight : array-like, optional
            Sample weights.
        class_weight_counts : dict, optional
            Precomputed {class_label: weight_sum}.

        Returns
        -------
        float
            Entropy impurity.
        """
        if class_weight_counts is not None:
            total = sum(class_weight_counts.values())
            if total == 0:
                return 0.0
            entropy = 0.0
            for w in class_weight_counts.values():
                p = w / total
                if p > 0:
                    entropy -= p * np.log2(p)
            return entropy

        proportions = compute_weighted_class_proportions(y, sample_weight)
        entropy = 0.0
        for p in proportions.values():
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

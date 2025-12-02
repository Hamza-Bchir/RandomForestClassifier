import numpy as np
from collections import Counter, defaultdict
from ._tree import TreeNode
from ._utils import Gini, Entropy, MisclassificationError, compute_weighted_class_proportions
from numpy.typing import ArrayLike

VALID_IMPURITY_MEASURES = {
    'gini': Gini,
    'entropy': Entropy,
    'misclassification_error': MisclassificationError
}

class DecisionTreeClassifier:
    """
    Custom weighted decision tree classifier supporting multiple impurity measures,
    feature subsampling, class weighting, and prediction probabilities.
    """

    def __init__(self,
                 impurity_measure='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weights=None,
                 random_state=None):
        """
        Initialize a decision tree classifier.
        """
        self.impurity_measure = impurity_measure
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weights = class_weights

        self.root = None
        self.classes_ = None
        self.n_features_in_ = None
        self.rng = np.random.default_rng(random_state)

    def _compute_sample_weight(self, y: ArrayLike):
        """
        Compute per-sample weights from class_weights.
        """
        n = len(y)
        if self.class_weights is None:
            return np.ones(n, dtype=float)
        elif self.class_weights == "balanced":
            counts = Counter(y)
            weight_for_class = {c: n / (len(counts) * counts[c]) for c in counts}
        elif isinstance(self.class_weights, dict):
            weight_for_class = self.class_weights
        else:
            raise ValueError("Invalid class_weights format.")
        return np.array([weight_for_class[c] for c in y], dtype=float)

    def _choose_feature_subset(self, n_features: int):
        """
        Subsample features according to max_features.
        """
        max_features = self.max_features
        if isinstance(max_features, int):
            k = max(1, min(n_features, max_features))
            return self.rng.choice(n_features, k, replace=False)
        if max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            return self.rng.choice(n_features, k, replace=False)
        if max_features == "log2":
            k = max(1, int(np.log2(n_features)))
            return self.rng.choice(n_features, k, replace=False)
        if max_features is None:
            return np.arange(n_features)
        raise ValueError("Invalid max_features")

    def _impurity_from_class_counts(self, counts_dict: dict):
        """
        Compute impurity given class weight counts.
        """
        imp_obj = VALID_IMPURITY_MEASURES[self.impurity_measure]()
        return imp_obj(class_weight_counts=counts_dict)

    def _find_bsplit(self, data, parent_imp: float):
        """
        Find the best split across the selected feature subset.
        Returns (feature_idx, threshold, gain, left_data, right_data).
        """
        X, y, sample_weight = data
        n_samples, n_features = X.shape
        if n_samples == 0:
            return None, None, 0.0, None, None

        classes = np.unique(y)
        self.classes_ = np.sort(classes) if self.classes_ is None else self.classes_

        total_counts = {c: float(np.sum(sample_weight[y == c])) for c in self.classes_}

        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        best_left = None
        best_right = None

        features_to_consider = self._choose_feature_subset(n_features)

        for feature_idx in features_to_consider:
            values = X[:, feature_idx]
            sort_idx = np.argsort(values)
            vals = values[sort_idx]
            y_sorted = y[sort_idx]
            w_sorted = sample_weight[sort_idx]

            cum_counts = {c: np.cumsum(w_sorted * (y_sorted == c)) for c in self.classes_}
            total_w = np.sum(w_sorted)
            unique_pos = np.nonzero(vals[1:] != vals[:-1])[0] + 1

            for pos in unique_pos:
                left_counts = {c: float(cum_counts[c][pos-1]) for c in self.classes_}
                right_counts = {c: float(total_counts[c] - left_counts[c]) for c in self.classes_}

                w_left = sum(left_counts.values())
                w_right = sum(right_counts.values())
                if w_left < 1e-12 or w_right < 1e-12:
                    continue
                if w_left < self.min_samples_leaf or w_right < self.min_samples_leaf:
                    continue

                left_imp = self._impurity_from_class_counts(left_counts)
                right_imp = self._impurity_from_class_counts(right_counts)
                weighted_imp = (w_left / total_w) * left_imp + (w_right / total_w) * right_imp
                gain = parent_imp - weighted_imp

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = float((vals[pos-1] + vals[pos]) / 2.0)
                    left_idx = sort_idx[:pos]
                    right_idx = sort_idx[pos:]
                    best_left = (X[left_idx], y[left_idx], sample_weight[left_idx])
                    best_right = (X[right_idx], y[right_idx], sample_weight[right_idx])

        return best_feature_idx, best_threshold, best_gain, best_left, best_right

    def _build_tree(self, data: tuple, current_depth: int = 0):
        """
        Recursively build the decision tree.
        """
        X, y, sample_weight = data
        n_samples = X.shape[0]

        if n_samples == 0:
            return TreeNode(data, prediction_proportions={}, feature_idx=None, threshold=None, impurity_gain=0.0)

        parent_counts = {c: float(np.sum(sample_weight[y == c])) for c in np.unique(y)}
        parent_imp = self._impurity_from_class_counts(parent_counts)

        if parent_imp == 0.0:
            return TreeNode(
                data=data,
                prediction_proportions=compute_weighted_class_proportions(y, sample_weight),
                feature_idx=None,
                threshold=None,
                impurity_gain=0.0
            )

        if self.max_depth is not None and current_depth >= self.max_depth:
            return TreeNode(
                data=data,
                prediction_proportions=compute_weighted_class_proportions(y, sample_weight),
                feature_idx=None,
                threshold=None,
                impurity_gain=0.0
            )

        if n_samples < self.min_samples_split:
            return TreeNode(
                data=data,
                prediction_proportions=compute_weighted_class_proportions(y, sample_weight),
                feature_idx=None,
                threshold=None,
                impurity_gain=0.0
            )

        feature_idx, threshold, gain, left_data, right_data = self._find_bsplit(data, parent_imp)

        if feature_idx is None or gain < self.min_impurity_decrease:
            return TreeNode(
                data=data,
                prediction_proportions=compute_weighted_class_proportions(y, sample_weight),
                feature_idx=None,
                threshold=None,
                impurity_gain=0.0
            )

        node = TreeNode(
            data=data,
            prediction_proportions=compute_weighted_class_proportions(y, sample_weight),
            feature_idx=feature_idx,
            threshold=threshold,
            impurity_gain=gain
        )

        node.left = self._build_tree(left_data, current_depth + 1)
        node.right = self._build_tree(right_data, current_depth + 1)
        return node

    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fit the decision tree classifier.
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        sample_weight = self._compute_sample_weight(y)
        self.root = self._build_tree((X, y, sample_weight))
        return self

    def _predict_one(self, x: ArrayLike, node: TreeNode):
        """
        Predict the class for a single sample.
        """
        if node.left is None and node.right is None:
            proportions = node.prediction_proportions
            if not proportions:
                return None
            max_prop = max(proportions.values())
            candidates = [c for c, p in proportions.items() if abs(p - max_prop) < 1e-12]
            return min(candidates)

        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: ArrayLike):
        """
        Predict class labels for samples.
        """
        X = np.array(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_proba_one(self, x: ArrayLike, node: TreeNode):
        """
        Predict class probability distribution for a single sample.
        """
        if node.left is None and node.right is None:
            prop = node.prediction_proportions
            return np.array([prop.get(c, 0.0) for c in self.classes_], dtype=float)

        if x[node.feature_idx] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        return self._predict_proba_one(x, node.right)

    def predict_proba(self, X: ArrayLike):
        """
        Predict class probabilities for samples.
        """
        X = np.array(X, dtype=np.float64)
        return np.vstack([self._predict_proba_one(x, self.root) for x in X])

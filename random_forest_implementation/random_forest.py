import numpy as np
from collections import Counter, defaultdict
from .decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="sqrt",   
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 class_weights=None,
                 random_state=None,
                 criterion="gini"):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.class_weights = class_weights
        self.random_state = random_state
        self.criterion = criterion
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []

    def _sample_bootstrap_indices(self, n_samples):
        return self.rng.integers(0, n_samples, size=n_samples)

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        n_samples, _ = X.shape

        self.trees_ = []

        for i in range(self.n_estimators):
            # bootstrap sample
            if self.bootstrap:
                idx = self._sample_bootstrap_indices(n_samples)
                X_sample = X[idx]
                y_sample = y[idx]

            else:
                X_sample, y_sample = X, y

            # create a tree that handles per-node max_features
            tree = DecisionTreeClassifier(
                impurity_measure=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,   # pass to tree; tree will subsample per node
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weights=self.class_weights,
                random_state=self.rng.integers(0, 1_000_000)
            )

            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if not self.trees_:
            raise ValueError("RandomForestClassifier not fitted")

        all_preds = np.array([t.predict(X) for t in self.trees_])  # (n_trees, n_samples)
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            counts = Counter(all_preds[:, i])
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            y_pred.append(min(candidates))  # deterministic tie-breaking
        return np.array(y_pred)

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        if not self.trees_:
            raise ValueError("RandomForestClassifier not fitted")
        # average tree probabilities (get tree.predict_proba)
        proba_sum = None
        for t in self.trees_:
            probs = t.predict_proba(X)  # shape (n_samples, n_classes) aligned to tree.classes_
            if proba_sum is None:
                proba_sum = probs
            else:
                proba_sum += probs
        return proba_sum / len(self.trees_)

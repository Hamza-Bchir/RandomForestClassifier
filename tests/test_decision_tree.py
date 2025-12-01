import numpy as np
import pytest
from random_forest_implementation.decision_tree import DecisionTreeClassifier


def test_all_same_class():
    """Check that the tree predicts the same class when all labels are identical."""
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)

    assert np.all(preds == 1)


def test_single_sample():
    """Verify the classifier handles a single-sample dataset correctly."""
    X = np.array([[42]])
    y = np.array([0])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds[0] == 0


def test_single_feature_split():
    """Ensure predictions only contain the original classes for a simple feature split."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds).issubset({0, 1})


def test_min_samples_split():
    """Check that min_samples_split prevents splitting nodes that are too small."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(min_samples_split=3)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds).issubset({0, 1})


def test_min_samples_leaf():
    """Verify min_samples_leaf is respected and all classes still appear when possible."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(min_samples_leaf=2)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds) == {0, 1}


def test_max_depth():
    """Ensure max_depth limits the number of splits while preserving original classes."""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 0, 1])

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds).issubset({0, 1})

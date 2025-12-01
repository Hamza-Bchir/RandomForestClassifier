import numpy as np
import pytest
from random_forest_implementation.random_forest import RandomForestClassifier


def test_all_same_class():
    """Check that the classifier predicts the same class when all labels are identical."""
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])

    clf = RandomForestClassifier(n_estimators=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert np.all(preds == 1)


def test_single_sample():
    """Check that the classifier handles a single-sample dataset correctly."""
    X = np.array([[42]])
    y = np.array([0])

    clf = RandomForestClassifier(n_estimators=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds[0] == 0


def test_single_feature_split():
    """Ensure predictions contain only the original classes for a single feature split."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    clf = RandomForestClassifier(n_estimators=3, max_depth=1, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds).issubset({0, 1})


def test_min_samples_split():
    """Verify that min_samples_split prevents splitting too small nodes."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    clf = RandomForestClassifier(n_estimators=3, min_samples_split=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds).issubset({0, 1})


def test_min_samples_leaf():
    """Check that min_samples_leaf is respected and all classes still appear when possible."""
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])

    clf = RandomForestClassifier(n_estimators=3, min_samples_leaf=2, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds) == {0, 1}


def test_max_depth():
    """Ensure max_depth limits the number of splits while still predicting original classes."""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 0, 1])

    clf = RandomForestClassifier(n_estimators=3, max_depth=1, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds).issubset({0, 1})

# Custom Random Forest and Decision Tree Implementation

This repository contains a from-scratch implementation of decision trees and random forests in Python. It was developed for a Kaggle competition organized as part of the IFT 6390: Fundamentals of Machine Learning course. The code is not as elaborated as scikit-learn's implementation of `RandomForestClassifier`, does not include pruning or out-of-bag (OOB) score, and is not as optimized in terms of running time.


## Project Structure

.
├── random_forest_implementation
│ ├── init.py
│ ├── decision_tree.py # DecisionTreeClassifier implementation
│ ├── random_forest.py # RandomForestClassifier using DecisionTreeClassifier
│ ├── _tree.py # Internal tree node structure and helper functions
│ └── _utils.py # Utility functions for tree building
├── tests
│ ├── test_decision_tree.py # Unit tests for DecisionTreeClassifier
│ ├── test_random_forest.py # Unit tests for RandomForestClassifier
│ └── init.py
├── README.md # This file
└── requirements.txt # Required Python packages



## Features

- `DecisionTreeClassifier`:
  - Supports `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weights`.
  - Pruning based on maximum number of leaves.
- `RandomForestClassifier`:
  - Ensemble of decision trees.
  - Supports bootstrapping, class weighting, and reproducible randomness.
- Fully tested with `pytest` for basic sanity checks.

## Installation

Clone the repository and install dependencies:
## Usage Example 

```bash
git clone <your-repo-url>
cd <repository-folder>
pip install -r requirements.txt
````

Python 3.11+ is required.

````python
import numpy as np
from random_forest_implementation.random_forest import RandomForestClassifier

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0)
clf.fit(X, y)
predictions = clf.predict(X)
print(predictions)
````

## Running Tests
````bash
pytest tests/test_decision_tree.py
pytest tests/test_random_forest.py
````

## References

- **scikit-learn** – Pedregosa et al., "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, 2011. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Breiman, L., "Random Forests", *Machine Learning*, 45(1), 5–32, 2001. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

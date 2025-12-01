from typing import Optional, Tuple
import numpy as np

class TreeNode:
    """
    Node of the Decision Tree.
    For leaf nodes, feature_idx and threshold are None.
    """
    def __init__(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        prediction_proportions: dict,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        impurity_gain: Optional[float] = None
    ) -> None:

        self.data = data
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.prediction_proportions = prediction_proportions
        self.impurity_gain = impurity_gain
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

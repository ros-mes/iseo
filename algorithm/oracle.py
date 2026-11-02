"""Oracle: membership oracle of an unknown knapsack constraint."""

import numpy as np


class Oracle:
    """Membership oracle of an unknown knapsack constraint ``w_bar @ x <= 1``.

    The oracle hides the weights of the constraint and only reveals whether a given point satisfies it.
    Every query is counted, so that the caller can enforce a query budget.

    Args:
        w_bar: Weights of the unknown knapsack constraint, shape ``(n,)``.

    Attributes:
        w_bar (np.ndarray): Weights of the unknown knapsack constraint, shape ``(n,)``.
        n_calls (int): Number of queries answered so far.
    """

    def __init__(self, w_bar: np.ndarray):

        self.w_bar = w_bar
        self.n_calls = 0

    def label(self, point: np.ndarray):
        """Labels a point with respect to the unknown constraint and counts the query.

        Args:
            point: Point to label, shape ``(n,)``.

        Returns:
            int: ``1`` (positive label) if ``w_bar @ point <= 1``, ``-1`` (negative label) otherwise.
        """

        self.n_calls += 1
        if np.dot(self.w_bar, point) <= 1:
            return 1
        else:
            return -1

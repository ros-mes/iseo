"""Oracle"""

import numpy as np


class Oracle:

    def __init__(self, w_bar: np.ndarray):

        self.w_bar = w_bar
        self.n_calls = 0

    def label(self, point: np.ndarray):

        self.n_calls += 1
        if np.dot(self.w_bar, point) <= 1:
            return 1
        else:
            return -1

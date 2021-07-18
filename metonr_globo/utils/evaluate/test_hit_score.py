from unittest import TestCase
import numpy as np

from utils.evaluate.eval_utils import hit_score


class TestHit_score(TestCase):
    def test_hit_score(self):
        a = np.zeros(20)
        a[0] = 1
        a[1] = 1
        y_true = a.astype(np.float)
        y_score = np.empty([20], dtype=float)
        x = hit_score(y_true, y_score)
        print(x)


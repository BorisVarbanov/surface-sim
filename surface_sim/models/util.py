import numpy as np


def num_biased_ops(n):
    inds = np.arange(n)
    res = np.sum(np.power(4, inds) * np.power(3, n - 1 - inds))
    return res

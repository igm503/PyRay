import numpy as np


def normalize(v: np.ndarray):
    return v / (np.linalg.norm(v) + 1e-6)

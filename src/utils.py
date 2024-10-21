import yaml

import numpy as np


def normalize(v: np.ndarray):
    return v / (np.linalg.norm(v) + 1e-6)


def load_yaml(yaml_path):
    with open(yaml_path, "r") as stream:
        content = yaml.safe_load(stream)
    return content

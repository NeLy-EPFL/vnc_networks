#!/usr/bin/env python3
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

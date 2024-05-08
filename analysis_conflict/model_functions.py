import numpy as np


def logistic_to_fraction(x):
    return 1 / (1 + np.exp(-x))


def logistic_to_percent(x):
    return 100 * logistic_to_fraction(x)

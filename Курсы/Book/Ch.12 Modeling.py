# 0 - столбик, 1 строка
# DEFAULT  ACROSS THE ROWS (for column), axis columns = ACROSS THE COLUMNS = for rows
from datetime import datetime

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=12345)


def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = (size,)
    return mean + np.sqrt(variance) * rng.standard_normal(*size)


N = 100

X = np.c_[dnorm(0, 0.4, size=N), dnorm(0, 0.6, size=N), dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]
y = np.dot(X, beta) + eps

# Other libraries from the book feel like they are deprecated

"""
This file defines helper function(s) to compute an "Equity" measure.

These measures include:

* Gini (https://en.wikipedia.org/wiki/Gini_coefficient)
* Hoover (https://en.wikipedia.org/wiki/Hoover_index)
"""

import numpy as np


def hoover(values):
    """
    Statistical measure of inequality.

    :param values: A list of integers, representing the distribution (incomes,
        comforts, ...).
    :return: A float between 0 (perfect equality) and 1 (perfect inequality).
    """
    if len(values) == 0:
        return 0.0
    mean = np.mean(values)
    sum_xi = 0
    sum_diff = 0
    for x in values:
        sum_xi += x
        sum_diff += abs(x - mean)
    return sum_diff / (2 * sum_xi + 10E-300)

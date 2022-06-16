import numpy as np


def interpolate(array, old_bounds, new_bounds):
    assert len(array) == len(old_bounds) == len(new_bounds)
    interpolated = [
        np.interp(array[k], old_bounds[k], new_bounds[k])
        for k in range(len(array))
    ]
    interpolated = np.array(interpolated)
    return interpolated

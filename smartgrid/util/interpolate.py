import numpy as np


def interpolate(value, old_bounds, new_bounds):
    """
    Interpolates a value (or array of values) from a domain to a new one.

    For example, if the value is ``0``, the interpolation from ``[-1, 1]`` to
    ``[0, 1]`` gives ``0.5``.

    This function is particularly useful for manipulating reward ranges, and
    to convert actions and observations between the ``[0, 1]`` domain that is
    typically used in learning algorithm (easier to manipulate) and the actual
    domain expected from the simulator.

    It supports interpolating differently for each dimension, when an array
    of values is passed.

    :param value: Either a single value (float) or an array of multiple values.
        It must match ``old_bounds`` and ``new_bounds``. The value(s) will be
        interpolated from their ``old_bounds`` to their ``new_bounds``.

    :param old_bounds: The previous domain (in which ``value`` is currently).
        If ``value`` is a scalar, ``old_bounds`` must be a 1D array of size 2,
        e.g., ``[-1, 1]``. Otherwise, if ``value`` is an array, ``old_bounds``
        must be a 2D array, of the same size as ``value``, each element being
        an array of size 2, e.g., ``[ [-1, 1], [0, 1], [-100, 100] ]`` assuming
        that ``value`` contains 3 elements.

    :param new_bounds: The new domain (in which the returned value will be).
        Similarly to ``old_bounds``, if ``value`` is a scalar, ``new_bounds``
        must be a 1D array of size 2, e.g., ``[0, 1]``. Otherwise, if ``value``
        is an array, ``new_bounds`` must be a 2D array, of the same size as
        ``value``, each element being an array of size 2, e.g., ``[ [0, 1],
        [-10, 10], [-1, 1] ]``, assuming that ``value`` contains 3 elements.

    :return: If ``value`` is a scalar (i.e., has no ``len()``), a scalar
        interpolated from the old domain to the new one. Otherwise, if ``value``
        is an array, a numpy ndarray is returned, in which each element was
        interpolated from its corresponding old domain to its new one.
    """
    try:
        size = len(value)
    except TypeError:
        # `value` has no length => it is a single value
        interpolated = np.interp(value, old_bounds, new_bounds)
        return interpolated
    # `value` has a length => array of multiple values
    assert size == len(old_bounds) == len(new_bounds)
    interpolated = [
        np.interp(value[k], old_bounds[k], new_bounds[k])
        for k in range(len(value))
    ]
    interpolated = np.array(interpolated)
    return interpolated

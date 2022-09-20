"""
This module defines operations on *bounded quantities*.

A *bounded quantity* is a quantity with a lower and/or upper limits (bounds).
Typically, it represents a physical quantity that realistically cannot go
beyond these bounds, e.g., a quantity that cannot be negative will have a
lower bound of ``0``.

Increasing and decreasing such quantities must take their bounds into account.
For example, a quantity that cannot be higher than 100: when increasing this
quantity by an amount, if the quantity + the amount exceeds 100, the rest will
be considered an *overhead*.
Similarly, when decreasing, if the quantity cannot be lower than 0, the
operation will return a *missing* quantity.

Examples: (assuming a lower bound of 0, and an upper bound of 100)

* 30 + 30 => new=60, overhead=0
* 60 + 50 => new=100, overhead=10
* 100 - 50 => new=50, missing=0
* 50 - 70 => new=0, missing=20

The `increase` and `decrease` functions return the following tuple:

* the new quantity, after the operation on the original quantity;
* the amount that was actually added or subtracted: may be lower than
  the intended amount, based on the constraints;
* the overhead or missing amount, i.e., the quantity that could not
  be added or subtracted. Note that the intended amount is equal to
  the actual amount + the overhead or missing amount.
"""


def increase_bounded(original_quantity, amount, upper_bound):
    """
    Increase a bounded quantity by an amount, enforcing an upper bound.

    :param original_quantity: The original quantity, before we increase it.
        It must be a number, e.g., int or float.
    :param amount: The amount we want to add to ``original_quantity``. The
        actual increase may be lower than this (because of the constraints),
        but cannot be higher. It must be a number, e.g., int or float, which
        can be added to the quantity (compatible types).
    :param upper_bound: The upper bound we want to enforce on the quantity.
        The new quantity is guaranteed to be lower or equal to this bound.

    :rtype: (float, float, float)
    :return:
        * **new_quantity** -- The new value of the quantity, after the increase.
        * **actual_amount** -- The actual amount that was added to the
          ``original_quantity``, i.e., ``original_quantity + actual_amount = new_quantity``.
        * **overhead** -- The quantity that could not be added (because it would
          have exceeded the bound), i.e., ``actual_amount + overhead = amount``.
    """
    assert amount >= 0
    # new_quantity is guaranteed to be <= upper_bound (since we use `min`)
    new_quantity = min(original_quantity + amount, upper_bound)
    # actual amount added (taking the upper bound into account)
    actual_amount = new_quantity - original_quantity
    # quantity that was not added
    overhead = amount - actual_amount
    return new_quantity, actual_amount, overhead


def decrease_bounded(original_quantity, amount, lower_bound):
    """
    Decrease a bounded quantity by an amount, enforcing a lower bound.

    :param original_quantity: The original quantity, before we decrease it.
        It must be a number, e.g., int or float.
    :param amount: The amount we want to subtract from ``original_quantity``.
        The actual decrease may be lower than this (because of the constraints),
        but cannot be higher. It must be a number, e.g., int or float, which
        can be subtracted to the quantity (compatible types).
    :param lower_bound: The lower bound we want to enforce on the quantity.
        The new quantity is guaranteed to be greater or equal to this bound.

    :rtype: (float, float, float)
    :return:
        * **new_quantity** -- The new value of the quantity, after the decrease.
        * **actual_amount** -- The actual amount that was subtracted to the
          ``original_quantity``, i.e., ``original_quantity - actual_amount = new_quantity``.
        * **missing** -- The quantity that could not be subtracted (because it would
          have exceeded the bound), i.e., ``actual_amount + missing = amount``.
    """
    assert amount >= 0
    # new_quantity is guaranteed to be >= lower_bound (since we use `max`)
    new_quantity = max(lower_bound, original_quantity - amount)
    # actual amount subtracted (taking lower bound into account)
    actual_amount = original_quantity - new_quantity
    # quantity that was not subtracted
    missing = amount - actual_amount
    # missing = max(0, amount - (original_quantity - lower_bound))
    return new_quantity, actual_amount, missing

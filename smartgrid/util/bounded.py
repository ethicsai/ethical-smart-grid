"""
This file defines operations on "bounded quantities".

For example, a quantity that cannot be higher than 100: when
increasing this quantity by an amount, if the quantity + the amount
exceeds 100, the rest will be considered an "overhead".

Similarly, when decreasing, if the quantity cannot be lower than 0,
the operation will return a "missing" quantity.

Examples: (assuming a lower bound of 0, and an upper bound of 100)
    30 + 30 => new=60, overhead=0
    60 + 50 => new=100, overhead=10
    100 - 50 => new=50, missing=0
    50 - 70 => new=0, missing=20

The `increase` and `decrease` functions return the following tuple:
 * the new quantity, after the operation on the original quantity
 * the amount that was actually added or subtracted: may be lower than
   the intended amount, based on the constraints
 * the overhead or missing amount, i.e., the quantity that could not
   be added or subtracted. Note that the intended amount is equal to
   the actual amount + the overhead or missing amount.
"""


def increase_bounded(original_quantity, amount, upper_bound):
    assert amount >= 0
    # new_quantity is guaranteed to be <= upper_bound (since we use `min`)
    new_quantity = min(original_quantity + amount, upper_bound)
    # actual amount added (taking the upper bound into account)
    actual_amount = new_quantity - original_quantity
    # quantity that was not added
    overhead = amount - actual_amount
    return new_quantity, actual_amount, overhead


def decrease_bounded(original_quantity, amount, lower_bound):
    assert amount >= 0
    # new_quantity is guaranteed to be >= lower_bound (since we use `max`)
    new_quantity = max(lower_bound, original_quantity - amount)
    # actual amount subtracted (taking lower bound into account)
    actual_amount = original_quantity - new_quantity
    # quantity that was not subtracted
    missing = amount - actual_amount
    # missing = max(0, amount - (original_quantity - lower_bound))
    return new_quantity, actual_amount, missing

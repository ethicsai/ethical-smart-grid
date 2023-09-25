"""
Judgment functions transform (activated) arguments into a scalar number.

The idea is to obtain a *reward*, which qualifies or measures the degree to
which the agent's action was aligned with the moral value that is represented
by the argumentation graph.
Argumentation frameworks allow determining a subset of *accepted* arguments,
such as the *grounded extension*. From this subset, we want to compare the
*pros* and *cons* arguments (which respectively defend that the action was
aligned or not aligned with the moral value), and to return a scalar number.

For example, if only *pros* arguments are present in the grounded extension,
the reward may be ``1``; if only *cons* arguments are present, the reward may
be ``0``. A mixture of *pros* and *cons* arguments should return a number
between ``0`` and ``1`` which corresponds to this mixture *and* which allows
the learning agent to effectively learn good behaviours.

The judgment functions implement this measure and propose several ways to
compute it, which have different properties.
A (hopefully good) discussion of these properties can be found at:
https://rchaput.github.io/phdthesis/5-judgments.html#judgments-experiments-ajar

This code is mostly based on the work of Benoît Alcaraz.
"""

from .afdm import AFDM


def j_simple(afdm: AFDM, decision: str):
    """
    Simply return the ratio of (alive) *pros* over *pros* + *cons*.
    """
    pros = len(afdm.arguments_supporting(decision))
    cons = len(afdm.arguments_countering(decision))
    if (pros + cons) == 0:
        # Neither moral nor immoral, we say it is neutral
        return 0.5
    else:
        return pros / (pros + cons)


def j_diff(afdm: AFDM, decision: str):
    """
    Compares the number of alive *pros* with total *pros*, and returns the
    difference with the number of alive *cons* over total *cons*.
    """
    alive_pros = len(afdm.arguments_supporting(decision))
    total_pros = len(afdm.arguments_supporting(decision, ignore_aliveness=True))
    alive_cons = len(afdm.arguments_countering(decision))
    total_cons = len(afdm.arguments_countering(decision, ignore_aliveness=True))

    pros = alive_pros / total_pros if total_pros != 0 else 0.0
    cons = alive_cons / total_cons if total_cons != 0 else 0.0

    return pros - cons


def j_ratio(afdm: AFDM, decision: str):
    """
    Compare the number of (squared) alive *pros* minus alive *cons* with the
    known maximum number of activated *pros* and *cons*.
    """
    # This function requires a "static" variable, to hold the known maximum
    # number. This number is updated each time the function is called.
    if not hasattr(j_ratio, 'best_acceptable_count'):
        j_ratio.best_acceptable_count = 0

    pros = len(afdm.arguments_supporting(decision))
    cons = len(afdm.arguments_countering(decision))

    top = (pros ** 2) - (cons ** 2)
    down = pros + cons

    # Update the known max
    j_ratio.best_acceptable_count = max(down, j_ratio.best_acceptable_count)
    if j_ratio.best_acceptable_count == 0:
        return 0.5
    else:
        return top / j_ratio.best_acceptable_count


def j_grad(afdm: AFDM, decision: str):
    """
    Create a gradient between 0 and 1, using as many graduations between 0.5
    and 1 as possible *pros* arguments, and as many graduations between 0.0 and
    0.5 as possible *cons* arguments. Then, advance graduations in a sense or
    another based on the number of alive *pros* and *cons*.
    """
    total_pros = len(afdm.arguments_supporting(decision, ignore_aliveness=True))
    total_cons = len(afdm.arguments_countering(decision, ignore_aliveness=True))
    # If there are no possible pros or cons (weird, but might happen), we simply
    # return 0.5
    if total_pros == 0 or total_cons == 0:
        return 0.5

    start = 0.5  # "neutral" point
    up_step = 0.5 / total_pros  # graduations towards 1
    down_step = 0.5 / total_cons  # graduations towards 0
    pros = len(afdm.arguments_supporting(decision))
    cons = len(afdm.arguments_countering(decision))
    return start + (up_step * pros) - (down_step * cons)


def j_offset(afdm: AFDM, decision: str):
    """
    Avoid division by 0 by offsetting the number of activated arguments.
    """
    pros = len(afdm.arguments_supporting(decision))
    cons = len(afdm.arguments_countering(decision))
    return min(1.0, (1 + pros) / (1 + cons))

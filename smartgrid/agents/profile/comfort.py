"""
Comfort functions determine the comfort of an agent from its consumption and need.

Each step of the simulation, an :py:class:`.Agent` has a need in terms of
energy to consume. When taking an :py:class:`.Action`, Agents decide how much
to consume (from the smart grid and their personal battery), to satisfy their
need. The degree to which this need is satisfied is called the **comfort**.

We recall that Agents' comforts are used in the :py:class:`.GlobalObservation`
as part of the *equity* measure, and in some reward functions (see
:py:mod:`smartgrid.rewards` for details). Intuitively, agents should have a
similar comfort to increase the equity.

To increase variety in the society, we propose that different *comfort functions*
may be used by different Agents. This allows controlling the effective comfort
of an Agent, based on its consumption and need.

For example, a simple function may return a comfort that linearly increase
with the ratio ``consumption / need``. Another function could have a different
slope, returning a very low comfort most of the time: for the same ratio
of consumption w.r.t. need, the 2nd function will return a lower comfort than
the 1st. Assuming that a reward function relying on equity is used, Agents
will try to achieve similar comforts. This means that an Agent using the 2nd
function would need a higher consumption to obtain the same comfort as an
Agent using the 1st function.

Thus, comfort functions can be used to differentiate Agents, and define some
sort of "priority" among them. Important buildings, such as Schools or
Hospitals, can use a comfort function that return lower comforts to ensure
their priority on energy consumption.
Comfort functions may also differentiate Agents of the same type, e.g.,
different Households where the inhabitants have different consumption habits,
preferences, etc.

We implement several comfort functions in this module; the simulator can be
extended to use other functions very easily, by implementing a new function
and providing it to the :py:class:`.AgentProfile`.
The only requirement is the signature: comfort functions take as input
the *consumption* and the *need*, and return the *comfort* (all floats).
However, please note that, whereas *consumption* and *need* are unbounded
(i.e., they can be any possible value in the real space), the *comfort*
must be in [0,1].

.. note::
    The following comfort functions are implemented using
    `Generalized logistic curves <https://en.wikipedia.org/wiki/Generalised_logistic_function>`_
    (similar to sigmoids), as they elegantly solve this requirement. In
    particular, they ensure that, even when ``consumption > need``, the
    resulting comfort will be capped at 1. They are also highly parameterizable.
    Yet, other comfort functions can use a different formula: this is not a
    requirement.

See the following image for a graphical representation of the resulting comfort
w.r.t. the consumption and need, for each of the implemented functions.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from smartgrid.agents.profile.comfort import (flexible_comfort_profile,
                                                  neutral_comfort_profile,
                                                  strict_comfort_profile)
    x = np.linspace(0, 1, 100)
    y1 = [flexible_comfort_profile(xi, 1) for xi in x]
    plt.plot(x, y1, label='flexible')
    y2 = [neutral_comfort_profile(xi, 1) for xi in x]
    plt.plot(x, y2, label='neutral')
    y3 = [strict_comfort_profile(xi, 1) for xi in x]
    plt.plot(x, y3, label='strict')
    plt.grid(visible=True, axis='both')
    plt.legend(title='Comfort function')
    plt.show()
"""

from decimal import Decimal


def flexible_comfort_profile(consumption: float, need: float) -> float:
    """
    Flexible comfort function: easy to satisfy.

    This function describes a curve that increases quickly.

    :param consumption: The quantity of energy consumed by an agent at a
        given time step.
    :param need: The quantity of energy that the agent needed at a given time
        step.
    :return: The agent's comfort, based on the ratio between consumption and
        need. The comfort is guaranteed to be within [0,1].
    """
    ratio = consumption / (need + 10E-300)
    return richard_curve(ratio, q=0.1, b=20, v=2, m=1 / 2)


def neutral_comfort_profile(consumption: float, need: float) -> float:
    """
    Neutral comfort function.

    This function describes a curve where the inflexion point, at y=0.5,
    is obtained when x=0.5.

    :param consumption: The quantity of energy consumed by an agent at a
        given time step.
    :param need: The quantity of energy that the agent needed at a given time
        step.
    :return: The agent's comfort, based on the ratio between consumption and
        need. The comfort is guaranteed to be within [0,1].
    """
    ratio = consumption / (need + 10E-300)
    return richard_curve(ratio, q=1, b=10, v=1, m=1 / 2)


def strict_comfort_profile(consumption: float, need: float) -> float:
    """
    Strict comfort function: difficult to satisfy.

    This function describes a curve that increases slowly.

    :param consumption: The quantity of energy consumed by an agent at a
        given time step.
    :param need: The quantity of energy that the agent needed at a given time
        step.
    :return: The agent's comfort, based on the ratio between consumption and
        need. The comfort is guaranteed to be within [0,1].
    """
    ratio = consumption / (need + 10E-300)
    return richard_curve(ratio, q=10, b=16, v=0.7, m=1 / 2)


def richard_curve(x, a=0.0, k=1.0, b=1.0, v=1.0, q=1.0, c=1.0, m=0.0) -> float:
    """
    Richard's Curve or Generalised logistic function.

    See https://en.wikipedia.org/wiki/Generalised_logistic_function for more
    details about the parameters.
    This function is used internally to build the *comfort functions*, by
    specifying various values for the parameters.

    :param x: The X value used to evaluate the function
    :param a: The lower asymptote
    :param k: The upper asymptote
    :param b: The growth rate
    :param v: Affects near which asymptote maximum growth occurs
    :param q: Related to the value of the curve at X=M (starting point, y0)
    :param c: Typically 1, otherwise it will shift the upper asymptote
    :param m: Starting point
    :return: The value of the curve at x.
    """
    a, k, b, v, q, c = (Decimal(a), Decimal(k), Decimal(b), Decimal(v), Decimal(q), Decimal(c))
    # x can be a numpy float, which is not directly convertible into Decimal
    x = Decimal(float(x))
    m = Decimal(float(m))
    return float(a + (k - a) / ((c + q * (-b * (x - m)).exp()) ** (1 / v)))

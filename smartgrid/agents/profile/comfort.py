from decimal import Decimal


def flexible_comfort_profile(consumption: float, need: float) -> float:
    ratio = consumption / need
    return richard_curve(ratio, q=0.1, b=20, v=2, m=1/2)


def neutral_comfort_profile(consumption: float, need: float) -> float:
    ratio = consumption / need
    return richard_curve(ratio, q=1, b=10, v=1, m=1/2)


def strict_comfort_profile(consumption: float, need: float) -> float:
    ratio = consumption / need
    return richard_curve(ratio, q=10, b=16, v=0.7, m=1/2)


def richard_curve(x, a=0.0, k=1.0, b=1.0, v=1.0, q=1.0, c=1.0, m=0.0) -> float:
    """
    Richard's Curve or Generalised logistic function.
    See https://en.wikipedia.org/wiki/Generalised_logistic_function

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

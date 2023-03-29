"""
(Learning) algorithms that can be used to learn policies for agents.

This package is separated from the smartgrid package, as the algorithms could be
used for other simulators (environments). Similarly, the smartgrid env could
be used with other learning algorithms.

Learning algorithms receive observations and rewards, and must decide which
actions should be taken, based on observations, such that the best rewards are
obtained.
"""

from .model import Model

"""
This package handles the Observations, which are information that the agent
knows about its environment. Observations are used by agents to assume in which
state they are, and, based on this, to decide which action should be taken.

In this Smart Grid environment, an Observation is a vector (list) of several
measures that are split into:

- **local** observations that are individual for each agent ;
- **global** observations that are common to all agents.

For example, the *hour* or *equity of comforts* are the same for all agents,
and can be shared: thus, they are deemed **global** observations.

On the other hand, the current amount of energy available in one's personal
battery, or one's current payoff, are individual measures: two agents may
very well have different observations for these measures. In addition, these
observations are *local*, which means that other agents cannot access them.
An agent only receives its *own* local observations, plus the global
observations that are shared by everyone.

The rationale behind this separation is that agents represent prosumers.
If this simulator was deployed in the real world, they would assist human
users by taking actions automatically for them, similarly to a domotic,
or home automation, system. There is therefore a privacy concern: we would
certainly not want our neighbour's agent to know our individual observations,
such as our payoff.

The :py:class:`.ObservationManager` is responsible for computing both the
:py:class:`.LocalObservation` from a single :py:class:`.Agent`, and the
:py:class:`.GlobalObservation` from the :py:class:`.World`.
They are then merged into :py:class:`.Observation`, which is sent to the
agents.

.. note::
    :py:class:`.Observation` does not differentiate between **local** and
    **global** observations: they are all merged into a single vector.
    Algorithms that require a distinction should therefore use
    :py:class:`.LocalObservation` and :py:class:`.GlobalObservation` instead.
"""

from .global_observation import GlobalObservation
from .local_observation import LocalObservation
from .observation_manager import ObservationManager
from .observations import Observation

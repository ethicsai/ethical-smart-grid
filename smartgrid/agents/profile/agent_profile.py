"""
AgentProfiles describe a few common characteristics of agents.
"""

from typing import Callable

import numpy as np
from gymnasium import spaces

from .need import NeedProfile
from .production import ProductionProfile


class AgentProfile:
    """
    Describes a few common characteristics of agents.

    Agents are separate entities, e.g., they have their own state, although
    some of their characteristics can be shared.
    For example, :py:class:`.Agent`\\ s have a need at each time step, i.e., a
    quantity of energy that they want to consume. Two different Agents, e.g.,
    two Households, may share the same need distribution, i.e., they draw
    their need from the same distribution.

    This is what we call the *profile* of an Agent. Intuitively, we say that
    *Household* is a profile, i.e., a set of common characteristics for all
    agents that correspond to a Household. *School* can be another profile,
    with different characteristics, etc.

    An *AgentProfile* is principally composed of the following characteristics:

    - ``name``: the profile's name, used for identification.
    - ``need_profile``: generator of needs for each time step.
    - ``production_fn``: generator of energy produced for each time step.
    - ``comfort_fn``: determines the comfort, based on need and consumption.
    - ``max_storage``: the capacity of the agent's personal battery.
    """

    name: str
    """The profile's name, used for identification."""

    action_space: spaces.Box
    """
    The space in which :py:class:`.Action`\\ s live.
    
    We briefly recall that Actions are composed of parameters for consuming
    and exchanging energy. Different *profiles* thus have different domains
    for these parameters, e.g., a *School* may consume more than a *Household*.
    It makes sense, as the *School* is typically a much bigger building than
    a *Household*. That is why the action space is determined by the *profile*
    (instead of fixed for all agents).
    
    It is represented as a Gym :py:class:`~gymnasium.spaces.Box`, which contains
    lower and upper bounds for each dimension of the :py:class:`.Action`.
    For example, a ``Box(0.0, 8500.0, (6,), int64)`` instance means that
    the action space comprises 6 dimensions, all of which are limited to
    the ``[0,8500]`` range, and values are int64 variables.
    
    .. note::
        It is common that algorithms produce actions as values between 0 and 1,
        as it is easier for neural networks. In this case, this ``action_space``
        can be useful for interpolating actions to their expected domain.
    
    .. note::
        We chose to use the same bounds (upper and lower) for all dimensions,
        although this is not a requirement. You may create a new *profile*
        with different bounds, e.g., if you want to restrict the amount
        of energy an agent can buy.
    """

    observation_space: spaces.Dict
    """The space in which :py:class:`.LocalObservation`\\ s live.
    
    Agents receive local observations that are individual to them. The profile
    determines the domain of these dimensions.
    """

    need_fn: NeedProfile
    """
    Generator of needs for each time step.
    
    At each time step, agents have a *need*, which is some sort of a target
    for the energy they want to consume. This need depends on their *profile*,
    e.g., a *School* will typically need more energy than a *Household*, and
    with a different "curve", e.g., most energy is consumed during the day for
    a *School*, whereas a *Household* will consume most energy in the evening.
    
    See :py:class:`.NeedProfile` for more details.
    """

    production_fn: ProductionProfile
    """
    Generator of energy produced for each time step.
    
    At each time step, agents produce a small quantity of energy for their
    own use. This production depends on their *profile*, e.g., a *School*
    have more surface than a *Household*, and can therefore display more
    photovoltaic (PV) panels, producing more energy.
    
    See :py:class:`.ProductionProfile` for more details.
    """

    comfort_fn: Callable[[float, float], float]
    """
    Function to determine the comfort level depending on consumption and need.
    
    Agents consume energy to satisfy their need; the degree to which they
    are satisfied is named the *comfort*, and is computed through this
    ``comfort_fn``. Different *profiles* may have different *comfort functions*,
    such that the comfort can be easier to obtain for some agents.
    
    For example, a *School* or a *Hospital* profile has a more important need,
    in the sense that a lack of consumption would result in problems. Thus,
    a small decrease of consumption should result in a large decrease of comfort.
    On contrary, other profiles could afford a small decrease of consumption
    as a small decrease of comfort.
    
    It is also possible to model different kind of prosumers, e.g., some people
    may accept to reduce their comfort when necessary, whereas others may be
    more "strict" and ask to consume what they need. In this example, several
    *Household* profiles could be derived, e.g., a *Flexible Household* or a
    *Strict Household*, etc.
    
    See the :py:mod:`~smartgrid.agents.profile.comfort` module for more details
    and implementations of *comfort functions*. Other comfort functions can be
    created, and used in *profiles*, provided that they respect the same
    signature.
    """

    max_storage: int
    """Maximum capacity of the agent's personal battery.
    
    Each agent has a personal battery in which they can store energy for later
    uses. The size (capacity) of this battery depends on the *profile*, e.g.,
    a *School* may possess a bigger battery, with a larger capacity, compared
    to a *Household*.
    """

    max_energy_needed: float
    """
    Maximum amount that agents of this profile can need at any time step.
    
    This is related to the :py:attr:`.need_fn` and is used particularly to
    determine the maximum amount that can be needed by all agents at any time
    step, in order to determine maximum bounds and normalize some observations
    to the ``[0,1]`` range (see :py:meth:`.EnergyGenerator.available_energy_bounds`
    and :py:meth:`.World.max_needed_energy` for more details).
    """

    def __init__(self,
                 name: str,
                 need_profile: NeedProfile,
                 production_profile: ProductionProfile,
                 max_storage: int,
                 action_space_low: np.ndarray,
                 action_space_high: np.ndarray,
                 action_dim: int,
                 comfort_fn):
        """
        Create an *AgentProfile*.

        :param name: The profile's name, for identification, e.g., *Household*.
            It is advised to use a human-friendly but still machine-usable
            name (avoid spaces, accents, etc.).
        :param need_profile: The NeedProfile that should be used in this profile.
        :param production_profile: The ProductionProfile that should be used
            in this profile.
        :param max_storage: The battery capacity of agents with this profile.
        :param action_space_low: The lower bounds of actions in this profile.
            It must be a 1D NumPy ndarray, with a value for each dimension.
            For example, ``np.asarray([0, 0, 0, 0, 0, 0])`` means that all
            dimensions will have a lower bound of ``0``. The bounds may also
            be different between dimensions, e.g., ``[0, 50, 30, 20, 60, 100]``.
        :param action_space_high: The higher bounds of actions in this profile.
            It must be a 1D NumPy ndarray, with a value for each dimension.
            For example, ``np.asarray([1000, 1000, 1000, 1000, 1000, 1000])``
            means that all dimensions will have a upper bound of ``1000``.
            The bounds may also be different between dimensions, e.g.,
            ``[1000, 1200, 800, 400, 2000, 1500]``.
        :param action_dim: Deprecated, unused.
        :param comfort_fn: The comfort function to use. See
            :py:mod:`smartgrid.agents.profile.comfort` for more details on
            comfort functions.
        """
        if action_space_low.shape != action_space_high.shape:
            raise Exception('action_space_low and action_space_high must '
                            'have the same shape! Found '
                            f'{action_space_low.shape} and '
                            f'{action_space_high.shape}')

        self.name = name

        # For easier access, we pre-compute the Action Space from low and high
        self.action_space = spaces.Box(
            low=action_space_low,
            high=action_space_high,
            shape=(action_dim,),
            dtype=int
        )

        # create observation_space for an agent
        self.observation_space = spaces.Dict({
            'personal_storage': spaces.Box(low=0, high=1, shape=(1,), dtype=int),
            'comfort': spaces.Box(0, 1, (1,), dtype=float),
            'payoff': spaces.Box(0, 1, (1,), dtype=float)
        })

        self.need_fn = need_profile
        self.production_fn = production_profile
        self.comfort_fn = comfort_fn

        self.max_storage = max_storage
        self.max_energy_needed = self.need_fn.max_energy_needed

    def need(self, step: int) -> float:
        """
        Generate a new need at a given time step for a single agent.

        :param step: The new time step.

        :return: The new value of need. Different agents may get different
            needs, even when using the same profile (depending on the
            implementation details of the :py:class:`NeedProfile`).
        """
        return self.need_fn.compute(step)

    def production(self, step: int) -> float:
        """
        Generate a new production at a given time step for a single agent.

        :param step: The new time step.

        :return: The new value of production. Different agents may get
            different needs, even when using the same profile (depending on
            the implementation details of the :py:class:`ProductionProfile`).
        """
        return self.production_fn.compute(step)

    def __str__(self):
        return '<AgentProfile name={}>'.format(self.name)

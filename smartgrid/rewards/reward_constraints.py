"""
RewardConstraints allow to activate a reward function only when certain
conditions are met.

Two intuitive type of constraints are:

*Time-based constraints*: enabling a reward function only during certain
time steps. This allows, for example, adding progressively new reward functions,
by enabling some only after a given number of time steps. This simulates a kind
of "evolving" ethical considerations, through this addition of reward functions.
See the :py:class:`.TimeConstrainedReward` for details.

*Agent-based constraints*: enabling a reward function only for certain agents.
This allows, for example, comparing the behaviour of two populations of
agents in the same simulation, driven by different objectives.
See the :py:class:`.AgentConstrainedReward` for details.

Constraints can be combined, e.g., a reward function can be both time-constrained
and agent-constrained, and extended, e.g., users can add their own constraints.
"""
from typing import Optional, List
from warnings import warn

from smartgrid.world import World
from smartgrid.agents import Agent
from smartgrid.rewards.reward import Reward


class TimeConstrainedReward(Reward):
    """
    Enable or disable a reward function based on the current time step.

    This constraint can be used to specify a *starting point*, before which
    the reward function is disabled and does not produce rewards, and an
    *ending point*, after which the reward function is disabled and does not
    produce rewards.

    This allows, for example, adding progressively reward functions: let us
    assume 3 reward functions A (always active), B (active after 10 steps), and
    C (active after 20 steps), learning agents will first only receive rewards
    from A, then from A and B, and finally from all three. This simulates
    the "evolution" of the ethical considerations that are embedded within the
    reward functions, as if this was a single reward whose definition changes
    along the time.
    """

    base_reward: Reward
    """
    The "base" reward function that we want to constrain.
    
    It is used to compute the reward when required.
    Note that it can be another type of constraint, so as to "combine"
    constraints as some sort of chain leading to the base reward function.
    """

    start_step: Optional[int]
    """
    Optional starting point of the reward, i.e., when the reward becomes enabled.

    This allows enabling/disabling rewards during a simulation, which creates
    changes in the environment, and forces agents to adapt to new (ethical)
    considerations.

    By default (``None``), the reward function is initially active, and produces
    rewards at the beginning of the simulation. If set to a (positive) integer,
    ``start_step`` places a constraint on the time steps at which the reward
    function becomes active: ``start_step <= t``, where ``t`` is the
    :py:attr:`~smartgrid.world.World.current_step`.

    See also :py:attr:`~.end_step` and :py:meth:`~.is_activated`.
    """

    end_step: Optional[int]
    """
    Optional end point of the reward, i.e., when the reward becomes disabled.

    This allows enabling/disabling rewards during a simulation, which creates
    changes in the environment, and forces agents to forget previous (ethical)
    considerations.

    By default (``None``), the reward function is never disabled after becoming
    active, and produces rewards at each time step. If set to a (positive)
    integer, ``end_step`` places a constraint on the time steps at which the
    reward function is active: ``t < end_step``, where ``t`` is the
    :py:attr:`~smartgrid.world.World.current_step`.

    See also :py:attr:`~.start_step` and :py:meth:`~.is_activated`.
    """

    def __init__(self,
                 base_reward: Reward,
                 start_step: int = None,
                 end_step: int = None):
        super().__init__(base_reward.name)
        self.base_reward = base_reward
        self.start_step = start_step
        self.end_step = end_step

        if end_step < start_step:
            warn(f'The ending step ({end_step} is before the starting step'
                 f' ({start_step}): the reward function ({base_reward.name})'
                 f' can never be activated!')

    def calculate(self, world: World, agent: Agent) -> float:
        return self.base_reward.calculate(world, agent)

    def is_activated(self, world: World, agent: Agent) -> bool:
        """
        Determines whether the reward function should produce a reward.

        In the ``TimeConstrainedReward``, it resorts to simply checking whether
        the world's :py:attr:`~smartgrid.world.World.current_step` lies between
        the :py:attr:`~.start_step` and the :py:attr:`~.end_step`.
        This allows:

        - enabling the reward function after a certain time, e.g.,
          ``start_step = 2000`` means that this reward function will only
          produce rewards from the 2000th time step;
        - disabling the reward function after a certain time, e.g.,
          ``end_step = 6000`` means that this reward function will only produce
          rewards before the 6000th time step;
        - mixtures of starting and ending times, with the constraint that
          ``start_step <= end_step`` (otherwise, the reward function cannot
          possibly be activated at any time step).

        :param world: The World in which the reward function may be activated;
            used primarily to obtain the current time step.
        :param agent: The Agent that should (potentially) be rewarded by this
            reward function; not used in this subclass, but required by the
            base signature.

        :return: A boolean indicating whether the reward function should
            produce a reward at this moment, based on the current time step.
        """
        # We have 3 conditions:
        # 1. The current time step must be after the starting point
        #    (or there is no starting point).
        # 2. The current time step must be before the ending point
        #    (or there is no ending point).
        # 3. The base reward must be enabled (this could be another constraint).
        # This order avoids unnecessary computations (in the base reward), if
        # the first condition is not met. As the base reward could be a
        # constraint itself, this could avoid lots of computations.
        step = world.current_step
        return (self.start_step is None or self.start_step <= step) and \
            (self.end_step is None or step < self.end_step) and \
            self.base_reward.is_activated(world, agent)

    def reset(self):
        self.base_reward.reset()

    def __str__(self):
        return 'TimeConstrainedReward<{};{};{}>'.format(
            self.start_step, self.end_step, str(self.base_reward)
        )

    __repr__ = __str__


class AgentConstrainedReward(Reward):
    """
    Enable or disable a reward function based on learning agents.

    This constraint can be used to specify which learning agents should receive
    rewards from a given reward function. For other learning agents, it is
    as if this reward function was not present.

    This allows, for example, training two populations of agents with different
    objectives: assume 4 agents [a1, a2, a3, a4] and 2 objectives A and B,
    reward function A can be constrained to only produce rewards for [a1, a2],
    whereas B can be constrained for [a3, a4]. This simulates a population
    of agents with heterogeneous ethical considerations (embedded within the
    reward functions).
    """

    base_reward: Reward
    """
    The "base" reward function that we want to constrain.
    
    It is used to compute the reward when required.
    Note that it can be another type of constraint, so as to "combine"
    constraints as some sort of chain leading to the base reward function.
    """

    agents: List[Agent]
    """
    List of agents that will receive rewards from this reward function.
    
    Other agents will not receive rewards from this function, as if it was
    disabled.
    """

    def __init__(self,
                 base_reward: Reward,
                 agents: List[Agent]):
        super().__init__(base_reward.name)
        self.base_reward = base_reward
        self.agents = agents

    def calculate(self, world: World, agent: Agent) -> float:
        return self.base_reward.calculate(world, agent)

    def is_activated(self, world: World, agent: Agent) -> bool:
        """
        Determines whether the reward function should produce a reward.

        In the ``AgentConstrainedReward``, it resorts to simply checking whether
        the learning agent is in the list of authorized agents
        :py:attr:`~.AgentConstrainedReward.agents`.

        :param world: The World in which the reward function may be activated;
            not used in this subclass, but required by the base signature.
        :param agent: The Agent that should (potentially) be rewarded by this
            reward function; compared to the list of authorized agents.

        :return: A boolean indicating whether the reward function should
            produce a reward at this moment, based on the learning agent.
        """
        # We have 2 conditions:
        # 1. The agent is in the list of "authorized" agents.
        # 2. The base reward is activated (this can be another constraint).
        # This order avoids unnecessary computations (in the base reward), if
        # the first condition is not met. As the base reward could be a
        # constraint itself, this could avoid lots of computations.
        return agent in self.agents and \
            self.base_reward.is_activated(world, agent)

    def reset(self):
        self.base_reward.reset()

    def __str__(self):
        return 'AgentConstrainedReward<{};{}>'.format(
            self.agents, str(self.base_reward)
        )

    __repr__ = __str__

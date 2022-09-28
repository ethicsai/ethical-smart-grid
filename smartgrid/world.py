from typing import List, Type

import numpy as np
from gym.vector.utils import spaces

from agents.agent import Agent
from observation.observation_manager import ObservationManager
from observation.observations import Observation
from rewards.reward import Reward
from rewards.reward_collection import RewardCollection
from util import EnergyGenerator


class World(object):
    """
    Represents the "physical" (simulated) smart grid.

    As per the Gym framework, our *World* represents the smart grid, and
    manages interactions within agents in the smart grid.
    It handles transitions between time steps, i.e., simulating the changes
    that happen, which are provoked by the agents' actions and the smart grid
    dynamics (energy generation, etc.).
    """

    agents: List[Agent]
    """
    List of all agents acting in the world.
    
    This list can be used to access the agents, which represent prosumers
    (buildings) and act in this world.
    
    There is (currently) no support for "scripted" agents, i.e., all agents
    in the world are considered to be "policy" agents: they receive observations
    and algorithms must decide actions for them, through the environment
    interaction loop (see :py:class:`.SmartGrid` for more details).
    """

    current_step: int
    """
    Current time step of the world.
    
    Each time step corresponds to a simulated hour, which the world keeps track
    of. The world is also responsible for incrementing the time steps, and
    simulating the next step.
    
    Initially, ``current_step`` is set to ``0``.
    """

    available_energy: int
    """
    Current quantity of available energy in the local grid.
    
    At each time step, the smart grid generates an important quantity of
    energy that is accessible to all agents.
    This quantity is assumed to come from an important but local source,
    such as a windmill farm, or an hydraulic power plant.
    
    It is separated from the *national grid*, which is considered unlimited,
    but must be paid by agents. On the contrary, the smart grid's *available
    energy* is free, but limited, and agents must learn not to consume too
    much so as to let energy for others.
    
    See :py:attr:`.energy_generator` for more details on how the *available
    energy* is generated at each step.
    """

    energy_generator: EnergyGenerator
    """
    Generator of energy for each time step, at the smart grid level.
    
    We recall that the smart grid locally generates an important quantity
    of energy accessible to all agents.
    
    In order to make this generation agnostic to the number and profiles
    of agents, the *energy generators* rely on the maximum energy needed
    by all agents.
    
    For example, consider 3 Households agents with a maximal need of 10kWh
    each. The maximum energy needed in the whole world will thus be 30kWh.
    A generator may produce between 80% and 120% of this maximal need, which
    means that in some cases, there is not enough energy for all agents, and
    they must reduce consumption (or risk preventing others from consuming),
    and in other cases, there is more energy than necessary, and they can store
    energy for later.
    
    See :py:class:`.EnergyGenerator` for more details.
    """

    observation_manager: ObservationManager
    """
    Responsible for generating Observations.
    
    The type of observations that are sent can be changed by specifying an
    instance of a subclass here. See also :py:attr:`.observation`.
    """

    observation: Type[Observation]
    """
    The type of Observations that are sent.
    
    This object (a class, not an instance) is used to get the shape of
    Observations, i.e., their domain.
    """

    reward_calculator: RewardCollection
    """
    Responsible for generating rewards.
    
    This class can be used to collect and aggregate several rewards functions, 
    as most algorithms expect a single (scalar) value.
    Reward functions may target different ethical considerations, see their
    implementations in :py:mod:`.rewards` for more details.
    """

    def __init__(self, observation_manager: ObservationManager, agents: List[Agent],
                 rewards: List[Reward], observation: Type[Observation], max_step: int,
                 energy_generator: EnergyGenerator):
        """
        Create a new simulated world.

        :param observation_manager: An instance of ObservationManager to
            create observations.
        :param agents: The list of agents that partake in this smart grid.
        :param rewards: The list of reward functions that are used.
        :param observation: The type (class) of Observations that are used.
        :param max_step: The maximum number of steps.
        :param energy_generator: The generator used to produce energy at
            each time step, based on the agents in the world and their needs.
        """
        self.observation = observation
        self.current_step = 0
        self.agents = agents
        self.observation_manager = observation_manager
        self.available_energy = 0
        self.energy_generator = energy_generator
        self.reward_calculator = RewardCollection(rewards)

        # calculate global_observation_space
        available_energy_per_step = [0] * max_step
        for a in self.agents:
            available_energy_per_step += a.profile.need_fn.need_per_hour

        low = min(available_energy_per_step) * self.energy_generator.lower
        high = max(available_energy_per_step) * self.energy_generator.upper

        global_space = {'available_energy': spaces.Box(low=low, high=high, shape=(1,), dtype=int),
                        'equity': spaces.Box(0, 1, (1,), dtype=np.float64),
                        'energy_loss': spaces.Box(0, 1, (1,), dtype=np.float64),
                        'autonomy': spaces.Box(0, 1, (1,), dtype=np.float64),
                        'exclusion': spaces.Box(0, 1, (1,), dtype=np.float64),
                        'well_being': spaces.Box(0, 1, (1,), dtype=np.float64),
                        'over_consumption': spaces.Box(0, 1, (1,), dtype=np.float64),
                        'hour': spaces.Box(0, 1, (1,), dtype=np.float64),
                        }

        # agglomerate local_observation_space
        local_space = {a.name: a.profile.observation_space for a in self.agents}
        space = {'local': spaces.Dict(local_space), 'global': spaces.Dict(global_space)}
        self.observation_space = spaces.Dict(space)

    def step(self):
        """
        Perform a new step of the simulation.

        This function performs the following:

        1. Actions are truly enacted. They were "intended" before, i.e., agents
        output a *decision*; now they are applied to the world, and the world
        is updated accordingly. For exemple by updating the agents' payoffs,
        their battery, the available energy, and so on.

        2. Agents are updated. They generate a new need, a new production,
        and they compute their comfort.

        3. A new ``available_energy`` is generated, based on the (new) agents'
        needs.
        """
        # Integrate all agents' actions
        for i, agent in enumerate(self.agents):
            agent.enacted_action = agent.handle_action()

        # Compute next state
        self.current_step += 1
        for agent in self.agents:
            agent.update(self.current_step)
        self.available_energy = self.energy_generator.compute_available_energy(sum([a.need for a in self.agents]))

    def reset(self):
        """
        Resets the state of the world to the initial state.

        This resets the current step, the observation manager, the agents
        themselves, and the available energy.

        This function must be called when initializing the world.
        """
        self.current_step = 0
        self.observation_manager.reset()
        for agent in self.agents:
            agent.reset()
        self.available_energy = self.energy_generator.compute_available_energy(sum([a.need for a in self.agents]))

    @property
    def max_needed_energy(self):
        """The total amount of energy that all agents need.

        It can be used for example to interpolate the current amount of
        available energy to [0,1].
        This maximum amount depends on the list of current agents,
        especially the maximum amount of energy that each may need.
        """
        return sum([agent.profile.max_energy_needed for agent in self.agents])

    def __str__(self):
        return '<World t={}>'.format(self.current_step)

    def get_info(self, reward_n):
        """
        Return additional information on the world (for the current time step).

        Information contain the rewards, for each agent.

        :param reward_n: The list of rewards, one for each agent.
        :return: A dict, containing an element with key ``rewards``.
            This element is itself a dict, indexed by the agents' names, and
            whose value is their reward.
        """
        info_n = {"rewards": {}}

        for i, agent in enumerate(self.agents):
            info_n["rewards"][agent.name] = reward_n[i]

        return info_n

    def get_observation_agent(self, agent):
        """
        Return the local observations for a single agent.

        See :py:class:`.ObservationManager` and :py:class:`.LocalObservation`
        for details.

        :param agent: The Agent for which we want to get local observations.
        :return: A LocalObservation instance corresponding to the agent's
            current state.
        :rtype: LocalObservation
        """
        return self.observation_manager.compute_agent(self, agent)

    def get_observation_global(self):
        """
        Return the global observations for the whole world.

        See :py:class:`.ObservationManager` and :py:class:`.GlobalObservation`
        for details.

        :return: A GlobalObservation instance corresponding to the world's
            current state, i.e., the entire society of agents.
        :rtype: GlobalObservation
        """
        return self.observation_manager.compute_global(self)

    @property
    def observation_shape(self):
        """
        The shape (i.e., domain) of observations.
        """
        return self.observation_manager.shape

    def get_reward(self, agent):
        """
        Compute and return the reward for an agent.

        This reward describes to which degree the agent's action was appropriate
        w.r.t. moral values. These moral values are encoded in the reward
        function, see :py:mod:`smartgrid.rewards` for details on reward functions.

        Reward functions may comprise multiple objectives. In such cases, they
        can be aggregated so that the result is a single float (which is used
        by most of the decision algorithms).
        This behaviour (whether to aggregate, and how to aggregate) is controlled
        by the :py:attr:`.reward_calculator`, see :py:class:`.RewardCollection`
        for details.
        """
        return self.reward_calculator.compute(self, agent)

from typing import List, Type

import numpy as np
from gym.vector.utils import spaces

from smartgrid.agents.agent import Agent
from smartgrid.observation.observation_manager import ObservationManager
from smartgrid.observation.observations import Observation
from smartgrid.rewards.reward import Reward
from smartgrid.rewards.reward_collection import RewardCollection
from smartgrid.util import EnergyGenerator


class World(object):
    # List of agents acting in the world
    # (currently, no distinction between policy and scripted agents)
    agents: List[Agent]
    # The current step, begins at 0. Represents the current time of simulation
    current_step: int
    # Quantity of available energy in the local grid
    # (that agents can freely access)
    available_energy: int
    observation_manager: ObservationManager
    observation: Type[Observation]
    energy_generator: EnergyGenerator

    def __init__(self, observation_manager: ObservationManager, agents: List[Agent],
                 rewards: List[Reward], observation: Type[Observation], max_step: int,
                 energy_generator: EnergyGenerator):
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

        global_space = {'available_energy': spaces.Box(low=low, high=high, shape=(1,), dtype=np.int),
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
        # Integrate all agents' actions
        for i, agent in enumerate(self.agents):
            agent.enacted_action = agent.handle_action()

        # Compute next state
        self.current_step += 1
        ratio_consprod = self._compute_ratio_consprod()
        for agent in self.agents:
            agent.update(self.current_step)
        self.available_energy = self.energy_generator.generate_available_energy(sum([a.need for a in self.agents]))

    def reset(self):
        self.current_step = 0
        self.observation_manager.reset()
        for agent in self.agents:
            agent.reset()
        self.available_energy = self.energy_generator.generate_available_energy(sum([a.need for a in self.agents]))

    def _compute_ratio_consprod(self):
        """Compute ratio between consumption and production."""
        # The initial quantity of available energy
        production = self.available_energy
        # The quantity consumed by agents
        consumption = sum([agent.enacted_action.grid_consumption
                           for agent in self.agents])
        return consumption / production

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
        info_n = {"rewards": {}}

        for agent in self.agents:
            info_n["rewards"][agent.name] = self.get_reward(agent)

        return info_n

    def get_observation_agent(self, agent):
        return self.observation_manager.compute_agent(self, agent)

    def get_observation_global(self):
        return self.observation_manager.compute_global(self)

    @property
    def observation_shape(self):
        return self.observation_manager.shape

    def get_reward(self, agent):
        return self.reward_calculator.compute(self, agent)

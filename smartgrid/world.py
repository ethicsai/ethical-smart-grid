from typing import List

from smartgrid.agents.agent import Agent, handle_action
from smartgrid.util import EnergyGenerator


class World(object):

    # List of agents acting in the world
    # (currently, no distinction between policy and scripted agents)
    agents: List[Agent]
    # `EnergyGenerator` to compute the next quantity of available energy
    energy_generator: EnergyGenerator
    # The current step, begins at 0. Represents the current time of simulation
    current_step: int
    # Quantity of available energy in the local grid
    # (that agents can freely access)
    available_energy: int

    def __init__(self, energy_generator):
        self.agents = []
        self.energy_generator = energy_generator
        self.current_step = 0
        self.available_energy = 0

    def step(self):
        # Integrate all agents' actions
        for i, agent in enumerate(self.agents):
            agent.enacted_action = handle_action(agent, agent.intended_action)

        # Compute next state
        self.current_step += 1
        ratio_consprod = self._compute_ratio_consprod()
        self.available_energy = self.energy_generator.generate_available_energy(self)
        for agent in self.agents:
            agent.update(self.current_step)

    def reset(self):
        self.current_step = 0
        self.available_energy = self.energy_generator.generate_available_energy(self)
        for agent in self.agents:
            agent.reset()

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
        return sum([agent.max_energy_needed for agent in self.agents])

    def __str__(self):
        return '<World t={}>'.format(self.current_step)

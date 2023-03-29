"""
The World represents the "physical" (simulated) smart grid.
"""

from typing import List

from smartgrid.agents import Agent
from smartgrid.util import EnergyGenerator


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

    def __init__(self,
                 agents: List[Agent],
                 energy_generator: EnergyGenerator):
        """
        Create a new simulated world.

        :param agents: The list of agents that partake in this smart grid.

        :param energy_generator: The generator used to produce energy at
            each time step, based on the agents in the world and their needs.
        """
        self.current_step = 0
        self.agents = agents
        self.available_energy = 0
        self.energy_generator = energy_generator

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
        # 1. Integrate all agents' actions
        for i, agent in enumerate(self.agents):
            agent.enacted_action = agent.handle_action()

        # 2. Update agents
        self.current_step += 1
        for agent in self.agents:
            agent.update(self.current_step)

        # 3. Generate new "available energy"
        self.available_energy = self.energy_generator.generate_available_energy(
            self.current_need,
            self.current_step,
            self.min_needed_energy,
            self.max_needed_energy
        )

    def reset(self):
        """
        Resets the state of the world to the initial state.

        This resets the current step, the observation manager, the agents
        themselves, and the available energy.

        This function must be called when initializing the world.
        """
        self.current_step = 0
        for agent in self.agents:
            agent.reset()
        self.available_energy = self.energy_generator.generate_available_energy(
            self.current_need,
            self.current_step,
            self.min_needed_energy,
            self.max_needed_energy
        )

    @property
    def current_need(self):
        """
        The total energy that agents currently need, for this time step.

        For now computed on-demand, but may be stored as an attribute in
        the future, to avoid computations.
        """
        return sum([a.need for a in self.agents])

    @property
    def min_needed_energy(self):
        """
        The minimum sum of energy that all agents need, for all time steps.

        Currently simplified to return 0, may be more accurate in the future.
        """
        return 0

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

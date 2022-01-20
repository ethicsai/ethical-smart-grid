import math
from typing import List

from smartgrid.agent import Action, Agent
from smartgrid.util.bounded import decrease_bounded, increase_bounded
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
            agent.enacted_action = self.handle_action(agent, agent.intended_action)

        # Compute next state
        self.current_step += 1
        ratio_consprod = self._compute_ratio_consprod()
        self.available_energy = self.energy_generator.generate_available_energy(self)
        for agent in self.agents:
            agent.state.comfort = agent.compute_comfort()
            agent.state.need = agent.compute_need(self.current_step)
            production = agent.compute_production(self.current_step)
            agent.increase_storage(production)

    def handle_action(self, agent: Agent, action: Action):
        # Temporary storage (without upper limit, but still a lower ;
        # we consider that energy is exchanged more or less at the
        # same instant)
        new_storage = agent.state.storage

        # 1. Agent buys energy
        # (may be limited by the current payoff)
        rate = 0.1
        price = math.ceil(rate * action.buy_energy)
        # limit price by current payoff
        agent.state.payoff, price, _ = decrease_bounded(agent.state.payoff,
                                                        price,
                                                        -1_000_000)
        # actually bought quantity
        bought = int(math.floor(price / rate))
        new_storage += bought

        # 2. Agent stores energy
        new_storage += action.store_energy

        # 3. Agent sells energy
        rate = 0.1
        new_storage, sold, _ = decrease_bounded(new_storage,
                                                action.sell_energy,
                                                0)
        price = math.floor(rate * sold)
        agent.state.payoff, _, _ = increase_bounded(agent.state.payoff,
                                                    price, 1_000_000)

        # 4. Agent consumes from storage
        # (may be limited by the storage)
        new_storage, storage_consumed, _ = decrease_bounded(new_storage,
                                                            action.storage_consumption,
                                                            0)

        # 5. Agent gives to the grid
        # (may be limited by the storage)
        new_storage, given, _ = decrease_bounded(new_storage,
                                                 action.give_energy,
                                                 0)

        # 6. Agent consumes from the grid
        # (we assume that agent can consume as much as wanted)
        grid_consumed = action.grid_consumption

        # Set the new storage
        # FIXME: ensure that new_storage is not > max_storage
        # perhaps decrease `store_energy` by the overhead?
        agent.state.storage = new_storage

        # We return the actually performed action (after application of
        # constraints), so we can log it
        action_enacted = Action(
            grid_consumption=grid_consumed,
            storage_consumption=storage_consumed,
            store_energy=action.store_energy,
            give_energy=given,
            buy_energy=bought,
            sell_energy=sold
        )
        return action_enacted

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

from collections import namedtuple

import numpy as np
from gym.spaces import Space

from smartgrid.util.bounded import (increase_bounded, decrease_bounded)


class AgentState(object):
    def __init__(self):
        self.comfort = 0
        self.payoff = 0
        self.storage = 0
        self.need = 0

    def __str__(self):
        return '<AgentState comfort={} payoff={} storage={} need={}'\
            .format(self.comfort, self.payoff, self.storage, self.need)


Action = namedtuple('Action', [
    'grid_consumption',
    'storage_consumption',
    'store_energy',
    'give_energy',
    'buy_energy',
    'sell_energy'
])


class Agent(object):

    name: str
    action_space: Space
    max_storage: int
    max_energy_needed: int
    state: AgentState
    intended_action: Action
    enacted_action: Action

    # The range in which the 'payoff' can be.
    payoff_range = (-10_000, +10_000)

    def __init__(self,
                 name: str,
                 action_space,
                 max_storage,
                 max_energy_needed,
                 compute_comfort,
                 compute_need,
                 compute_production,
                 ):

        # Constant attributes
        self.name = name
        self.action_space = action_space
        self.max_storage = max_storage
        self.max_energy_needed = max_energy_needed

        # Callbacks (for parametrizing agent profiles)
        self._compute_comfort = compute_comfort
        self._compute_need = compute_need
        self._compute_production = compute_production

        # State and action are updated throughout the simulation
        self.reset()

    def increase_storage(self, amount):
        new, added, overhead = increase_bounded(self.state.storage,
                                                amount,
                                                self.max_storage)
        self.state.storage = new
        return new, added, overhead

    def decrease_storage(self, amount):
        new, subtracted, missing = decrease_bounded(self.state.storage,
                                                    amount,
                                                    0)
        self.state.storage = new
        return new, subtracted, missing

    def compute_comfort(self):
        # Total quantity consumed by agent
        consumption = self.enacted_action.grid_consumption \
                      + self.enacted_action.storage_consumption
        # Energy that agent needed
        need = self.state.need
        return self._compute_comfort(consumption, need)

    def compute_need(self, current_step):
        return self._compute_need(current_step)

    def compute_production(self, current_step):
        return self._compute_production(current_step)

    def reset(self):
        self.state = AgentState()
        self.state.need = self._compute_need(0)
        self.intended_action = Action(*[0] * len(Action._fields))
        self.enacted_action = self.intended_action

    @property
    def storage_ratio(self):
        """Return the current storage quantity over its capacity (in [0,1])."""
        return self.state.storage / (self.max_storage + 10E-300)

    @property
    def payoff_ratio(self):
        """Return the current payoff scaled to [0,1]"""
        return np.interp(self.state.payoff, Agent.payoff_range, (0, 1))

    def __str__(self):
        return '<Agent {}>'.format(self.name)

    __repr__ = __str__

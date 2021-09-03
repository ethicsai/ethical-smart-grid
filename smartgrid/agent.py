from collections import namedtuple
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
    state: AgentState
    action: Action

    def __init__(self,
                 name: str,
                 action_space,
                 max_storage,
                 compute_comfort,
                 compute_need,
                 compute_production,
                 ):

        # Constant attributes
        self.name = name
        self.action_space = action_space
        self.max_storage = max_storage

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
        consumption = self.action.grid_consumption \
                      + self.action.storage_consumption
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
        self.action = Action(*[0] * len(Action._fields))

    def __str__(self):
        return '<Agent {}>'.format(self.name)

    __repr__ = __str__

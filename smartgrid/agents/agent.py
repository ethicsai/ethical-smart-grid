import math
from collections import namedtuple

import numpy as np

from .profile import AgentProfile
from smartgrid.util.bounded import (increase_bounded, decrease_bounded)


class AgentState(object):
    def __init__(self):
        self.comfort = 0
        self.payoff = 0
        self.storage = 0
        self.need = 0

    def __str__(self):
        return '<AgentState comfort={} payoff={} storage={} need={}' \
            .format(self.comfort, self.payoff, self.storage, self.need)

    def reset(self):
        self.__init__()


Action = namedtuple('Action', [
    'grid_consumption',
    'storage_consumption',
    'store_energy',
    'give_energy',
    'buy_energy',
    'sell_energy'
])


class Agent(object):
    """
    An agent represent the physical entity in the world. He contains:
     - name for differencing
     - the state of our Agent
     - the max battery storage possible
     - an intended action for a step, this represents the purpose of an Actor.
     - an enacted action for a step, this represents the realist action.
    """
    name: str
    state: AgentState
    intended_action: Action
    enacted_action: Action
    profile: AgentProfile

    # The range in which the 'payoff' can be.
    payoff_range = (-10_000, +10_000)

    def __init__(self,
                 name: str,
                 profile: AgentProfile,
                 ):
        # Constant attributes
        self.name = name

        # Callbacks (for parametrizing agent profiles)
        self.profile = profile
        self.state = AgentState()

        # State and action are updated throughout the simulation
        self.reset()

    def increase_storage(self, amount: float) -> (float, float, float):
        """
        Function for adding some energy in the storage.
        :param amount: energy for charging the battery.
        :returns: a tuple of float with the quantity in the battery,
                  the energy added and the energy that cannot be stocked.
        """
        new, added, overhead = increase_bounded(self.state.storage,
                                                amount,
                                                self.profile.max_storage)
        self.state.storage = new
        return new, added, overhead

    def decrease_storage(self, amount: float) -> (float, float, float):
        """
        Function for adding some energy in the storage.
        :param amount: energy for charging the battery.
        :returns: a tuple of float with the quantity in the battery, the energy took and the energy that missed.
        """
        new, subtracted, missing = decrease_bounded(self.state.storage,
                                                    amount,
                                                    0)
        self.state.storage = new
        return new, subtracted, missing

    def update(self, step: int) -> None:
        """
        Function for updating all metric for our agent
        :param step: the current step
        """
        consumption = self.enacted_action.grid_consumption + self.enacted_action.storage_consumption
        self.profile.update(step)
        self.increase_storage(self.profile.production)
        self.state.comfort = self.profile.comfort_fn(consumption, self.need)

    def reset(self):
        self.state.reset()
        self.profile.reset()
        self.state.need = self.need
        self.intended_action = Action(*[0.0] * len(Action._fields))
        self.enacted_action = self.intended_action

    @property
    def need(self):
        return self.profile.need

    @property
    def production(self):
        return self.profile.production

    @property
    def comfort(self):
        return self.state.comfort

    @property
    def storage_ratio(self) -> float:
        """Return the current storage quantity over its capacity (in [0,1])."""
        return self.state.storage / (self.profile.max_storage + 10E-300)

    @property
    def payoff_ratio(self) -> float:
        """Return the current payoff scaled to [0,1]"""
        return np.interp(self.state.payoff, Agent.payoff_range, (0, 1))

    def __str__(self):
        return '<Agent {}>'.format(self.name)

    __repr__ = __str__

    def handle_action(self) -> Action:
        """
        handle_action is used to transform an intended_action into an enacted_action.
        It performs some computation for updating state of our Agent.
        """
        # Temporary storage (without upper limit, but still a lower ;
        # we consider that energy is exchanged more or less at the
        # same instant)
        action = self.intended_action
        new_storage = self.state.storage

        # 1. Agent buys energy
        # (may be limited by the current payoff)
        rate = 0.1
        price = math.ceil(rate * action.buy_energy)
        # limit price by current payoff
        self.state.payoff, price, _ = decrease_bounded(self.state.payoff,
                                                       price,
                                                       -1_000_000)
        # actually bought quantity
        bought = int(math.floor(price / rate))
        new_storage += bought

        # 2. Agent stores energy
        # TODO see with Rémy if this need to be bounded (max cap on battery)
        new_storage += action.store_energy

        # 3. Agent sells energy
        rate = 0.1
        new_storage, sold, _ = decrease_bounded(new_storage,
                                                action.sell_energy,
                                                0)
        price = math.floor(rate * sold)
        self.state.payoff, _, _ = increase_bounded(self.state.payoff,
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

        # Increase the storage consumption by the overflow on battery
        if new_storage > self.profile.max_storage:
            storage_consumed += self.profile.max_storage - new_storage
            new_storage = self.profile.max_storage

        # Set the new storage
        self.state.storage = new_storage

        # We return the actually performed action (after application of
        # constraints), so we can log it
        action_enacted = Action(
            grid_consumption=float(grid_consumed),
            storage_consumption=float(storage_consumed),
            store_energy=float(action.store_energy),
            give_energy=float(given),
            buy_energy=float(bought),
            sell_energy=float(sold)
        )
        return action_enacted

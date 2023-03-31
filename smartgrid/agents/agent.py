"""
This module contains the Agent class, its state (AgentState) and Action.
"""

import math
from collections import namedtuple

import numpy as np

from .profile import AgentProfile
from smartgrid.util.bounded import (increase_bounded, decrease_bounded)


class AgentState(object):
    """
    The (mutable) state of an Agent.
    """

    comfort: float
    """
    The agent's current comfort, a float that *should* be in [0,1].
    """

    payoff: float
    """
    The agent's current payoff, i.e., the cumulated sum of benefits and losses.
    
    .. note::
        The payoff should be within the 
        :py:attr:`smartgrid.agents.agent.Agent.payoff_range` .
    """

    storage: float
    """
    The agent's current amount of energy stored in its personal battery.
    """

    need: float
    """
    The agent's current need, i.e., energy that it would like to consume.
    """

    production: float
    """
    The agent's energy produced at the current step, and put in its storage.
    
    .. note::
        As the storage is limited, it may happen that the difference between
        the new storage and the storage at the previous step is smaller than
        the `production`.
    """

    def __init__(self):
        self.comfort = 0
        self.payoff = 0
        self.storage = 0
        self.need = 0
        self.production = 0

    def __repr__(self):
        return '<AgentState comfort={} payoff={} storage={} need={} production={}' \
            .format(self.comfort, self.payoff, self.storage, self.need, self.production)

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
Action.__doc__ = """
    An immutable (named) tuple containing action parameters.
    
    Actions may be either *intended* (i.e., what the learning algorithm or
    agent's policy would like to do) or *enacted* (i.e., what truly happened
    considering the physical constraints of the world).
"""


class Agent(object):
    """
    An agent represents a physical entity in the world.

    It contains:
     - a (unique) name for identifying the agent;
     - a (current) state;
     - an intended action for a step, what the Agent wanted to do;
     - an enacted action for a step, what the Agent really did;
     - an agent profile, the common characteristics shared by multiple agents.
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

        # Agent profile (contains "callbacks" to compute needs, productions, ...)
        self.profile = profile
        # The agent's (current) state, updated every time step in `update`.
        self.state = AgentState()
        # Note: the (intended/enacted) actions are initialized in `reset` to
        # avoid setting them twice.

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

        :returns: a tuple of float with the quantity in the battery, the energy
            took and the energy that was missing.
        """
        new, subtracted, missing = decrease_bounded(self.state.storage,
                                                    amount,
                                                    0)
        self.state.storage = new
        return new, subtracted, missing

    def update(self, step: int) -> None:
        """
        Update the agent's current state (production, need, storage, comfort).

        :param step: The current time step.
        """
        # Compute comfort (using the previous need)
        consumption = self.enacted_action.grid_consumption + self.enacted_action.storage_consumption
        self.state.comfort = self.profile.comfort_fn(consumption, self.state.need)

        # Compute a new need
        self.state.need = self.profile.need(step)

        # Compute a new production and increase the storage accordingly
        self.state.production = self.profile.production(step)
        self.increase_storage(self.state.production)

    def reset(self):
        # Reset all state values to 0
        self.state.reset()

        # Create a fake action (0 for all parameters)
        self.intended_action = Action(*[0.0] * len(Action._fields))
        self.enacted_action = self.intended_action

        # Update state for the 1st step (need, production, storage, ...)
        # Note that comfort will most likely remain at 0 since action does nothing
        self.update(0)

    @property
    def need(self):
        return self.state.need

    @property
    def production(self):
        return self.state.production

    @property
    def comfort(self):
        return self.state.comfort

    @property
    def storage_ratio(self) -> float:
        """Return the current storage quantity over its capacity (in [0,1])."""
        return self.state.storage / (self.profile.max_storage + 10E-300)

    @property
    def payoff_ratio(self) -> float:
        """Return the current payoff scaled to [0,1]."""
        return np.interp(self.state.payoff, self.payoff_range, (0, 1))

    def __str__(self):
        return '<Agent {}>'.format(self.name)

    __repr__ = __str__

    def handle_action(self) -> Action:
        """
        Perform the intended action and transform it into the enacted action.

        The *intended* action represents the action the agent intends to do,
        if possible, but it may happen that, due to some constraint, e.g.,
        battery capacity, it is not possible as-is.
        This method thus transforms the *intended* action into an *enacted*
        action, taking into account these constraints, and updates the Agent's
        state according to the *enacted* action.

        :return: The *enacted* action that truly happened, after the Agent's
            state was updated.
        """
        # Temporary storage (without upper limit, but still a lower ;
        # we consider that energy is exchanged more or less at the
        # same instant). This allows, e.g., to buy more energy than the capacity
        # allows, and to instantly consume this energy.
        # For example, assuming that the max storage is 500Wh, we can buy 1000Wh,
        # immediately consume 500Wh, give 150Wh, and store the remaining 350Wh.
        action = self.intended_action
        new_storage = self.state.storage

        # 1. Agent buys energy
        # (can be limited by the current payoff)
        rate = 0.1
        price = math.ceil(rate * action.buy_energy)
        # limit price by current payoff
        self.state.payoff, price, _ = decrease_bounded(self.state.payoff,
                                                       price,
                                                       self.payoff_range[0])
        # actually bought quantity
        bought = int(math.floor(price / rate))
        new_storage += bought

        # 2. Agent stores energy
        # (agent can store as much as desired)
        new_storage += action.store_energy

        # 3. Agent sells energy
        # (can be limited by the current storage, including bought and stored)
        # Note: agent could sell for more than it can really gain, because the
        # payoff is bounded. In this case, the money is "lost".
        rate = 0.1
        new_storage, sold, _ = decrease_bounded(new_storage,
                                                action.sell_energy,
                                                0)
        price = math.floor(rate * sold)
        self.state.payoff, _, _ = increase_bounded(self.state.payoff,
                                                   price,
                                                   self.payoff_range[1])

        # 4. Agent consumes from storage
        # (can be limited by the storage)
        new_storage, storage_consumed, _ = decrease_bounded(new_storage,
                                                            action.storage_consumption,
                                                            0)

        # 5. Agent gives to the grid
        # (can be limited by the storage)
        new_storage, given, _ = decrease_bounded(new_storage,
                                                 action.give_energy,
                                                 0)

        # 6. Agent consumes from the grid
        # (we assume that agent can consume as much as wanted)
        grid_consumed = action.grid_consumption

        # At this point, the new storage can be greater than the capacity
        # (overflow). This is by design (see comment at the top of the method),
        # but we now need to fix this. Possible ways include:
        # - Overflow energy could be wasted ("disappear") => not realistic.
        # - Reduce the quantity of stored energy, and/or bought energy => which
        #   should we reduce first? Is it possible that we cannot reduce them
        #   enough? What happens in this case?
        # - Increase the storage consumption => seems more realistic (energy
        #   *must* be consumed), and easier to implement than reducing other
        #   parameters. => This is the chosen way to deal with overflow.
        if new_storage > self.profile.max_storage:
            storage_consumed += (new_storage - self.profile.max_storage)
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

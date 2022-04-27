from abc import ABC, abstractmethod

import numpy as np

from smartgrid.agents.agent import Agent
from smartgrid.util import hoover
from smartgrid.world import World


# TODO herit from nameTuple
# TODO see if update is needed
# TODO suppress herit
class GlobalObservation(ABC):
    """
    All observation of the World. It's the same for all Agent in the grid.
    Compute once by step.
    It contains:
        - hour: represent by 24 values
        - available_energy: in the grid
        - equity: 1 - hoover of the Comfort
        - energy_loss: in the grid
        - autonomy: metric that represent the self sustain of the grid
        - exclusion: mean of Agent where their comfort is inferior well_being/2
        - well_being: median of the Comfort
        - over_consumption: energy that was present physically in the world
        - sum_taken: energy take by all Agent in the Grid
    """
    hour: float
    available_energy: float
    equity: float
    energy_loss: float
    autonomy: float
    exclusion: float
    well_being: float
    over_consumption: float
    sum_taken: float

    # computation reduction
    last_step_compute: int
    pass

    def _is_compute(self, world: World)->bool:
        return world.current_step == self.last_step_compute

    @abstractmethod
    def compute(self, world: World)->None:
        pass

    @abstractmethod
    def update(self, world: World, agent: Agent):
        pass

    def reset(self):
        self.last_step_compute = -1


class BaseGlobal(GlobalObservation):
    def __init__(self):
        self.last_step_compute = -1

    def update(self, world: World, agent: Agent):
        # todo implement it
        pass

    def compute(self, world: World):
        # return directly if the step have been computed
        if self._is_compute(world):
            return

        # Pre-compute some intermediate data
        comforts = []
        self.sum_taken, sum_given, sum_transactions, sum_consumed, sum_stored = 0, 0, 0, 0, 0
        for a in world.agents:
            comforts.append(a.state.comfort)
            self.sum_taken += a.enacted_action.grid_consumption \
                              + a.enacted_action.store_energy
            sum_given += a.enacted_action.give_energy
            sum_transactions += a.enacted_action.buy_energy \
                                + a.enacted_action.sell_energy
            sum_consumed += a.enacted_action.grid_consumption \
                            + a.enacted_action.storage_consumption
            sum_stored += a.enacted_action.store_energy

        # Compute some common measures about env
        self.hour = (world.current_step % 24) / 24
        self.available_energy = np.interp(world.available_energy,
                                          world.energy_generator.available_energy_bounds(world),
                                          (0, 1))
        self.equity = 1.0 - hoover(comforts)

        self.over_consumption = max(0.0, self.sum_taken - sum_given - world.available_energy)
        self.over_consumption /= (self.sum_taken + 10E-300)

        self.energy_loss = max(0.0, -self.over_consumption)

        self.autonomy = 1.0 - sum_transactions / (sum_consumed + sum_stored
                                                  + sum_given + sum_transactions + 10E-300)

        self.well_being = np.median(comforts)
        if np.isnan(self.well_being):
            self.well_being = 0.0

        threshold = self.well_being / 2
        self.exclusion = len([c for c in comforts if c < threshold]) / len(comforts)

        self.last_step_compute = world.current_step

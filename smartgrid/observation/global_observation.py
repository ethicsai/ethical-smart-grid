from abc import abstractmethod
from collections import namedtuple

import numpy as np

from smartgrid.agents.agent import Agent
from smartgrid.util import hoover



# TODO see if update is needed
class GlobalObservation(namedtuple('GlobalObservation', global_fields)):
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

    # computation reduction
    last_step_compute = -1

    @classmethod
    def _is_compute(self, world: 'World') -> bool:
        return world.current_step == self.last_step_compute

    @classmethod
    def compute(cls, world: World):
        # return directly if the step have been computed
        if cls._is_compute(world):
            return cls.computed

        # Pre-compute some intermediate data
        comforts = []
        sum_taken, sum_given, sum_transactions, sum_consumed, sum_stored = 0, 0, 0, 0, 0
        for a in world.agents:
            comforts.append(a.state.comfort)
            sum_taken += a.enacted_action.grid_consumption \
                                     + a.enacted_action.store_energy
            sum_given += a.enacted_action.give_energy
            sum_transactions += a.enacted_action.buy_energy \
                                + a.enacted_action.sell_energy
            sum_consumed += a.enacted_action.grid_consumption \
                            + a.enacted_action.storage_consumption
            sum_stored += a.enacted_action.store_energy

        # Compute some common measures about env
        hour = (world.current_step % 24) / 24
        available_energy = np.interp(world.available_energy,
                                                 world.energy_generator.available_energy_bounds(world),
                                                 (0, 1))
        equity = 1.0 - hoover(comforts)

        over_consumption = max(0.0, sum_taken - sum_given - world.available_energy)
        over_consumption /= (sum_taken + 10E-300)

        energy_loss = max(0.0, -over_consumption)

        autonomy = 1.0 - sum_transactions / (sum_consumed + sum_stored
                                                         + sum_given + sum_transactions + 10E-300)

        well_being = np.median(comforts)
        if np.isnan(well_being):
            well_being = 0.0

        threshold = well_being / 2
        exclusion = len([c for c in comforts if c < threshold]) / len(comforts)

        cls.last_step_compute = world.current_step
        cls.computed = cls(hour=hour,
                    available_energy=available_energy,
                    equity=equity,
                    energy_loss=energy_loss,
                    autonomy=autonomy,
                    exclusion=exclusion,
                    well_being=well_being,
                    over_consumption=over_consumption,
        )
        return cls.computed

    @abstractmethod
    def update(self, world: 'World', agent: Agent):
        # todo implement it
        pass

    @classmethod
    def reset(cls):
        cls.last_step_compute = -1

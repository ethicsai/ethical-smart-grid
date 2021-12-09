from __future__ import annotations
from collections import namedtuple

import numpy as np

from smartgrid.util.equity import hoover

fields = [
    'hour',
    'available_energy',
    'personal_storage',
    'comfort',
    'payoff',
    'equity',
    'energy_loss',
    'autonomy',
    'exclusion',
    'well_being',
    'over_consumption',
]


class Observation(namedtuple('Observation', fields)):

    def check_between_0_and_1(self):
        errors = {k: v for k, v in self._asdict().items() if v < 0.0 or v > 1.0}
        if len(errors) > 0:
            import warnings
            warnings.warn('Incorrect observations, not in [0,1]: {}'.format(errors),
                          stacklevel=2)

    @classmethod
    def compute(cls, env: SmartGrid, agent: Agent):
        """Compute observations that an agent will receive about the env."""

        # Pre-compute some intermediate data
        comforts = []
        sum_taken, sum_given, sum_transactions, sum_consumed, sum_stored = 0, 0, 0, 0, 0
        for a in env.agents:
            comforts.append(a.state.comfort)
            sum_taken += a.enacted_action.grid_consumption \
                         + a.enacted_action.store_energy
            sum_given += a.enacted_action.give_energy
            sum_transactions += a.enacted_action.buy_energy \
                                + a.enacted_action.sell_energy
            sum_consumed += a.enacted_action.grid_consumption \
                            + a.enacted_action.storage_consumption
            sum_stored += a.enacted_action.store_energy

        # Individual data
        personal_storage = agent.storage_ratio
        comfort = agent.state.comfort
        payoff = agent.payoff_ratio

        # Compute some common measures about env
        hour = (env.world.current_step % 24) / 24
        available_energy = env.world.available_energy  # FIXME: scale to [0,1]!
        equity = 1.0 - hoover(comforts)

        over_consumption = max(0, sum_taken - sum_given - env.world.available_energy)
        over_consumption /= (sum_taken + 10E-300)

        energy_loss = max(0, -over_consumption)

        autonomy = 1.0 - sum_transactions / (sum_consumed + sum_stored
                                             + sum_given + sum_transactions + 10E-300)

        well_being = np.median(comforts)
        if np.isnan(well_being):
            well_being = 0.0

        threshold = well_being / 2
        exclusion = len([c for c in comforts if c < threshold]) / len(comforts)

        obs = cls(hour=hour,
                  available_energy=available_energy,
                  personal_storage=personal_storage,
                  comfort=comfort,
                  payoff=payoff,
                  equity=equity,
                  energy_loss=energy_loss,
                  autonomy=autonomy,
                  exclusion=exclusion,
                  well_being=well_being,
                  over_consumption=over_consumption)
        # obs.check_between_0_and_1()

        return obs

    def __array__(self) -> np.ndarray:
        # Using `[*values()]` seems more efficient than other methods
        # e.g., `list(values())` or `values()` directly.
        return np.array([*self._asdict().values()])

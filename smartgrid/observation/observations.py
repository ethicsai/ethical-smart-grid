from __future__ import annotations

from collections import namedtuple

import numpy as np

from smartgrid.observation.global_observation import GlobalObservation
from smartgrid.observation.local_observation import LocalObservation

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


#TODO herit from nameTuple
#TODO suppress herit

class Observation(namedtuple('Observation', fields)):

    @classmethod
    def get_observation_space(cls):
        return fields

    def check_between_0_and_1(self):
        errors = {k: v for k, v in self._asdict().items() if v < 0.0 or v > 1.0}
        if len(errors) > 0:
            import warnings
            warnings.warn('Incorrect observations, not in [0,1]: {}'.format(errors),
                          stacklevel=2)

    def __array__(self) -> np.ndarray:
        # Using `[*values()]` seems more efficient than other methods
        # e.g., `list(values())` or `values()` directly.
        return np.array([*self._asdict().values()])

    @classmethod
    def create(cls, local_observation: LocalObservation, global_observation: GlobalObservation):
        obs = cls(hour=global_observation.hour,
                  available_energy=global_observation.available_energy,
                  personal_storage=local_observation.personal_storage,
                  comfort=local_observation.comfort,
                  payoff=local_observation.payoff,
                  equity=global_observation.equity,
                  energy_loss=global_observation.energy_loss,
                  autonomy=global_observation.autonomy,
                  exclusion=global_observation.exclusion,
                  well_being=global_observation.well_being,
                  over_consumption=global_observation.over_consumption)
        obs.check_between_0_and_1()

        return obs

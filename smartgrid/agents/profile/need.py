from abc import abstractmethod
from typing import List


class NeedProfile:
    """
    The need profile is the energy that an entity need for a step.
    It contains:
        - all `need_per_hour` from dataConversion
        - max_need from all data. It's used for creating the observation_space
    """
    need_per_hour: List[float]
    max_energy_needed: float

    def __init__(self, need_per_hour):
        self.need_per_hour = need_per_hour
        self.max_energy_needed = max(need_per_hour)

    @abstractmethod
    def compute(self, step=0) -> float:
        step %= len(self.need_per_hour)
        return self.need_per_hour[step]

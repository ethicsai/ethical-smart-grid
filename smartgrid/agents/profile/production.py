from typing import List


class ProductionProfile:
    """
    The production profile is the energy product by an entity for a step.
    It contains:
        - all `production_per_hour` from dataConversion
        - production for a step (remember of compute methods)
    """
    production_per_hour: List[float]

    def __init__(self, production_per_hour):
        self.production_per_hour = production_per_hour

    def compute(self, step=0) -> float:
        step %= len(self.production_per_hour)
        return self.production_per_hour[step]

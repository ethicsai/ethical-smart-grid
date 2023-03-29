"""
Per-agent reward functions directly look at the agent itself (and its action)
to determine its contribution.
"""

from .adaptability import (AdaptabilityOnePerAgent,
                           AdaptabilityTwoPerAgent,
                           AdaptabilityThreePerAgent)

from .comfort import Comfort

from .equity import EquityPerAgent

from .multi_objective_sum import MultiObjectiveSumPerAgent

from .over_consumption import OverConsumptionPerAgent

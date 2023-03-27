from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.per_agent.equity import EquityPerAgent
from smartgrid.rewards.numeric.per_agent.multi_objective_sum import MultiObjectiveSumPerAgent
from smartgrid.rewards.numeric.per_agent.over_consumption import OverConsumptionPerAgent
from smartgrid.rewards.reward import Reward


class AdaptabilityOnePerAgent(Reward):
    """
    Equity when t<3000, MultiObjectiveSum otherwise.

    This reward function changes its definition after time step t=3000.
    With t < 3000, it performs exactly as the :py:class:`.EquityPerAgent` reward
    function. When t >= 3000, it performs as the :py:class:`.MultiObjectiveSumPerAgent`
    reward function, which is a weighted average of the :py:class:`.Comfort`
    and :py:class:`.OverConsumptionPerAgent`.

    Thus, the targeted objectives are completely different in the two phases
    (equity vs comfort+overconsumption).
    This makes this reward function useful to evaluate whether agents are
    able to "completely" change their behaviour.
    """

    name: str

    def __init__(self):
        super().__init__()
        self.equity = EquityPerAgent()
        self.mos = MultiObjectiveSumPerAgent()

    def calculate(self, world, agent) -> float:
        if world.current_step < 3000:
            return self.equity.calculate(world, agent)
        else:
            return self.mos.calculate(world, agent)


class AdaptabilityTwoPerAgent(Reward):
    """
    Equity when t<2000, (Equity+OverConsumption)/2 otherwise.

    This reward function changes its definition after time step t=2000.
    With t < 2000, it performs exactly as the :py:class:`.EquityPerAgent` reward
    function. When t >= 2000, it returns the average of :py:class:`.EquityPerAgent`
    and the :py:class:`.OverConsumptionPerAgent` reward functions.

    Thus, the targeted objectives increase in the second phase: the initial one
    is kept, and a new one is added (equity vs equity+overconsumption).
    This makes this reward function useful to evaluate whether agents are
    able to change their behaviour by taking into account new objectives
    in addition to previous ones.

    This reward function is easier than :py:class:`.AdaptabilityOnePerAgent`
    (which completely replace the set of objectives) and
    :py:class:`.AdaptabilityThreePerAgent` (which uses 3 phases instead of 2).
    """

    name: str

    def __init__(self):
        super().__init__()
        self.equity = EquityPerAgent()
        self.over_consumption = OverConsumptionPerAgent()

    def calculate(self, world, agent) -> float:
        if world.current_step < 2000:
            return self.equity.calculate(world, agent)
        else:
            equity = self.equity.calculate(world, agent)
            oc = self.over_consumption.calculate(world, agent)
            return (equity + oc) / 2


class AdaptabilityThreePerAgent(Reward):
    """
    Equity when t<2000, (Equity+OverConsumption)/2 when t<6000, (Equity+OC+Comfort)/3 otherwise.

    This reward function changes its definition after time step t=2000 and
    after t=6000. With t < 2000, it performs exactly as the :py:class:`.EquityPerAgent`
    reward function. When 2000 <= t < 6000, it returns the average of
    :py:class:`.EquityPerAgent` and :py:class:`.OverConsumptionPerAgent`.
    Finally, when t >= 6000, it returns the average of :py:class:`.EquityPerAgent`,
    :py:class:`.OverConsumptionPerAgent`, and :py:class:`.Comfort`.

    Thus, the targeted objectives increase in the second and third phases: the
    previous ones are kept, and a new one is added.
    This makes this reward function useful to evaluate whether agents are
    able to change their behaviour by taking into account new objectives
    in addition to previous ones.
    """

    name: str

    def __init__(self):
        super().__init__()
        self.equity = EquityPerAgent()
        self.over_consumption = OverConsumptionPerAgent()
        self.comfort = Comfort()

    def calculate(self, world, agent):
        equity = self.equity.calculate(world, agent)
        if world.current_step < 2000:
            return equity
        elif world.current_step < 6000:
            oc = self.over_consumption.calculate(world, agent)
            return (equity + oc) / 2
        else:
            oc = self.over_consumption.calculate(world, agent)
            comfort = self.comfort.calculate(world, agent)
            return (equity + oc + comfort) / 3

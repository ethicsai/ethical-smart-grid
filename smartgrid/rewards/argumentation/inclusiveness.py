"""
Reward based on the *Inclusiveness* moral value, promoting equity of comforts.
"""

from smartgrid.rewards import Reward
from smartgrid.rewards.argumentation.situation import parse_situation
from ajar import (
    JudgingAgent, judgment, AFDM, Argument
)


class Inclusiveness(Reward):
    """
    Reward based on the *Inclusiveness* moral value, promoting equity of comforts.
    """

    name: str

    judge: JudgingAgent
    """
    The JudgingAgent that represents the implemented moral value.
    
    It contains the argumentation graph (:py:class:`ajar.afdm.AFDM`) that stores
    the arguments and attacks between them, as well as the transform function,
    and performs the judgment over a given situation.
    """

    decision: str = 'moral'
    """
    Hardcoded decision to use for the judgment.
    
    The AFDM framework that we use supports multiple decisions; in this current
    version we only need to decide whether the action was *moral* or not.
    Thus, we use an hardcoded identifier for the decision: it must be the same
    when creating :py:class:`ajar.argument.Argument`\\s and when performing the
    judgment!
    """

    def __init__(self, j_transform=None):
        super().__init__()
        if j_transform is None:
            j_transform = judgment.j_simple
        afdm = self._create_afdm()
        self.judge = JudgingAgent('inclusiveness', afdm, j_transform)

    def calculate(self, world, agent):
        situation = parse_situation(world, agent)
        return self.judge.judge(situation, self.decision)

    def _create_afdm(self):
        af = AFDM()

        af.add_argument(Argument(
            "c>>moy_1", "comfort more than 10 points above average",
            lambda s: s["self_comfort"] - s["average_comfort"] >= 0.10,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "c>>moy_2", "comfort more than 15 points above average",
            lambda s: s["self_comfort"] - s["average_comfort"] >= 0.15,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "c>moy", "comfort more than 5 points above average",
            lambda s: s["self_comfort"] - s["average_comfort"] >= 0.05,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "school", "agent is a school",
            lambda s: s["agent_type"] == "School"
        ))

        af.add_argument(Argument(
            "give_energy_1", "agent gave energy more than 10% of its need",
            lambda s: s["give_energy"] >= s["self_need"] * 0.1,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "give_energy_2", "agent gave energy more than 20% of its need",
            lambda s: s["give_energy"] >= s["self_need"] * 0.2,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "give_energy_3", "agent gave energy more than 30% of its need",
            lambda s: s["give_energy"] >= s["self_need"] * 0.3,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "c<moy", "comfort below more than 5 points of average",
            lambda s: s["self_comfort"] - s["average_comfort"] <= -0.05
        ))

        af.add_argument(Argument(
            "c<<moy", "comfort below more than 10 points of average",
            lambda s: s["self_comfort"] - s["average_comfort"] <= -0.10,
            counter=[self.decision]
        ))

        af.add_attack_relation("school", "c>moy")
        af.add_attack_relation("c>>moy_1", "school")
        af.add_attack_relation("c>>moy_2", "school")

        af.add_attack_relation("c<moy", "give_energy_1")
        af.add_attack_relation("c<moy", "give_energy_2")
        af.add_attack_relation("c<moy", "give_energy_3")

        af.add_attack_relation("give_energy_2", "c>moy")
        af.add_attack_relation("give_energy_3", "c>moy")

        return af

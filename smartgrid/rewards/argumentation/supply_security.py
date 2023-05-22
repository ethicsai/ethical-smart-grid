"""
Reward based on the *security of supply* moral value, promoting well-being.
"""

from smartgrid.rewards import Reward
from smartgrid.rewards.argumentation.situation import parse_situation
from ajar import (
    JudgingAgent, judgment, AFDM, Argument
)


class SupplySecurity(Reward):
    """
    Reward based on the *security of supply* moral value, promoting well-being.
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

    LOW_GLOBAL = 130_000
    HIGH_GLOBAL = 190_000

    def __init__(self, j_transform=None):
        super().__init__()
        if j_transform is None:
            j_transform = judgment.j_simple
        afdm = self._create_afdm()
        self.judge = JudgingAgent('supply_security', afdm, j_transform)

    def calculate(self, world, agent):
        situation = parse_situation(world, agent)
        return self.judge.judge(situation, self.decision)

    def _create_afdm(self):
        af = AFDM()

        af.add_argument(Argument(
            "good_comfort_1", "comfort above 94%",
            lambda s: s["self_comfort"] > 0.94,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "good_comfort_2", "comfort above 96%",
            lambda s: s["self_comfort"] > 0.96,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "average_comfort", "comfort above 91%",
            lambda s: s["self_comfort"] > 0.91,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "bad_comfort_1", "comfort less than 91%",
            lambda s: s["self_comfort"] < 0.91,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "bad_comfort_2", "comfort less than 85%",
            lambda s: s["self_comfort"] < 0.85,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "bad_comfort_3", "comfort less than 75%",
            lambda s: s["self_comfort"] < 0.75,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "bias_1", "bias",
            lambda s: True,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "bias_2", "bias",
            lambda s: True,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "high_global", "global available energy is high",
            lambda s: s["global_energy"] > self.HIGH_GLOBAL
        ))

        af.add_argument(Argument(
            "low_global", "global available energy is low",
            lambda s: s["global_energy"] < self.LOW_GLOBAL
        ))

        af.add_argument(Argument(
            "school", "agent is a school",
            lambda s: s["agent_type"] == "School"
        ))

        af.add_argument(Argument(
            "over_consume_1", "agent over-consumed more than 10% of its need",
            lambda s: s["self_need"] * 1.10 < s["consumption"],
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "over_consume_2", "agent over-consumed more than 20% of its need",
            lambda s: s["self_need"] * 1.20 < s["consumption"],
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "over_consume_3", "agent over-consumed more than 30% of its need",
            lambda s: s["self_need"] * 1.30 < s["consumption"],
            counter=[self.decision]
        ))

        af.add_attack_relation("high_global", "average_comfort")
        af.add_attack_relation("school", "high_global")
        af.add_attack_relation("low_global", "bad_comfort_1")

        af.add_attack_relation("school", "average_comfort")

        af.add_attack_relation("school", "over_consume_1")

        af.add_attack_relation("over_consume_1", "good_comfort_1")
        af.add_attack_relation("over_consume_2", "good_comfort_1")
        af.add_attack_relation("over_consume_3", "good_comfort_1")

        af.add_attack_relation("over_consume_1", "good_comfort_2")
        af.add_attack_relation("over_consume_2", "good_comfort_2")
        af.add_attack_relation("over_consume_3", "good_comfort_2")

        af.add_attack_relation("good_comfort_2", "bias_2")
        af.add_attack_relation("bad_comfort_3", "bias_1")

        return af

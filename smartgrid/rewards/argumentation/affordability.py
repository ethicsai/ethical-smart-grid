"""
Reward based on the *affordability* moral value, promoting not paying too much.
"""

from smartgrid.rewards import Reward
from smartgrid.rewards.argumentation.situation import parse_situation
from ajar import (
    JudgingAgent, judgment, AFDM, Argument
)


class Affordability(Reward):
    """
    Reward based on the *affordability* moral value, promoting not paying too much.
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
        self.judge = JudgingAgent('affordability', afdm, j_transform)

    def calculate(self, world, agent):
        situation = parse_situation(world, agent)
        return self.judge.judge(situation, self.decision)

    def _create_afdm(self):
        af = AFDM()

        af.add_argument(Argument(
            "good_comfort", "comfort above 98%",
            lambda s: s["self_comfort"] >= 0.98
        ))

        af.add_argument(Argument(
            "average_comfort", "comfort above 91%",
            lambda s: s["self_comfort"] > 0.91
        ))

        af.add_argument(Argument(
            "positive_payoff", "money earned by buying/selling energy is positive",
            lambda s: s["payoff"] >= 0,
            support=[self.decision]
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

        af.add_argument(Argument(
            "buy_energy_1", "agent bought energy more than 10% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.1,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_2", "agent bought energy more than 30% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.3,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_3", "agent bought energy more than 50% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.5,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_4", "agent bought energy more than 75% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.75,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "not_buy_energy", "agent bought energy less than 10% of its need",
            lambda s: s["buy"] <= s["self_need"] * 0.1,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "bias", "bias when agent did not buy",
            lambda s: True,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "bad_comfort", "comfort less than 90%",
            lambda s: s["self_comfort"] < 0.9
        ))

        af.add_attack_relation("bad_comfort", "positive_payoff")
        af.add_attack_relation("bad_comfort", "not_buy_energy")
        af.add_attack_relation("average_comfort", "buy_energy_1")

        af.add_attack_relation("over_consume_1", "good_comfort")
        af.add_attack_relation("over_consume_2", "good_comfort")
        af.add_attack_relation("over_consume_3", "good_comfort")

        af.add_attack_relation("good_comfort", "buy_energy_1")
        af.add_attack_relation("good_comfort", "buy_energy_2")
        af.add_attack_relation("good_comfort", "buy_energy_3")
        af.add_attack_relation("good_comfort", "buy_energy_4")

        af.add_attack_relation("buy_energy_4", "bias")

        af.add_attack_relation("positive_payoff", "buy_energy_1")
        af.add_attack_relation("positive_payoff", "buy_energy_2")
        af.add_attack_relation("positive_payoff", "buy_energy_3")
        af.add_attack_relation("positive_payoff", "buy_energy_4")

        return af

"""
Reward based on the *environmental sustainability* moral value, promoting ecology.
"""

from smartgrid.rewards import Reward
from smartgrid.rewards.argumentation.situation import parse_situation
from ajar import (
    JudgingAgent, judgment, AFDM, Argument
)


class EnvironmentalSustainability(Reward):
    """
    Reward based on the *environmental sustainability* moral value, promoting
    ecology.
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
        self.judge = JudgingAgent('env_sustain', afdm, j_transform)

    def calculate(self, world, agent):
        situation = parse_situation(world, agent)
        return self.judge.judge(situation, self.decision)

    def _create_afdm(self):
        af = AFDM()

        af.add_argument(Argument(
            "bad_comfort", "comfort lower than 95%",
            lambda s: s["self_comfort"] < 0.95,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "cons_storage", "consumed from storage more than 50% of capacity",
            lambda s: s["storage_consumption"] > 0.5 * s["storage_capacity"],
        ))

        af.add_argument(Argument(
            "over_consume", "agent over-consumed more than 10% of its need",
            lambda s: s["self_need"] * 1.1 < s["consumption"]
        ))

        af.add_argument(Argument(
            "bias", "this is a bias",
            lambda s: True,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "bias_2", "this is a bias",
            lambda s: True,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "bias_3", "this is a bias",
            lambda s: True,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_1", "agent bought energy more than 15% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.15,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_2", "agent bought energy more than 25% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.25,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_3", "agent bought energy more than 40% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.4,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "buy_energy_4", "agent bought energy more than 55% of its need",
            lambda s: s["buy"] > s["self_need"] * 0.55,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "stored_1", "agent storage is filled more than 50% of capacity",
            lambda s: s["storage_n"] / s["storage_capacity"] > 0.5,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "stored_2", "agent storage is filled more than 80% of capacity",
            lambda s: s["storage_n"] / s["storage_capacity"] > 0.8,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "stores_1", "agent stored energy more than 25% of capacity",
            lambda s: s["delta_store"] / s["storage_capacity"] > 0.25,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "stores_2", "agent stored energy more than 50% of capacity",
            lambda s: s["delta_store"] / s["storage_capacity"] > 0.5,
            support=[self.decision]
        ))

        af.add_argument(Argument(
            "high_global", "global available energy is high",
            lambda s: s["global_energy"] > self.HIGH_GLOBAL,
            counter=[self.decision]
        ))

        af.add_argument(Argument(
            "low_global", "global available energy is high",
            lambda s: s["global_energy"] < self.LOW_GLOBAL,
        ))

        af.add_attack_relation("bad_comfort", "bias")
        af.add_attack_relation("cons_storage", "bad_comfort")
        # af.add_attack_relation("cons_storage", "buy_energy")
        af.add_attack_relation("over_consume", "cons_storage")

        af.add_attack_relation("stored_1", "high_global")
        af.add_attack_relation("stores_1", "high_global")
        af.add_attack_relation("high_global", "bias")
        af.add_attack_relation("stores_2", "bias_2")
        af.add_attack_relation("stored_2", "bias_2")
        af.add_attack_relation("low_global", "bias_2")

        af.add_attack_relation("buy_energy_1", "stores_1")
        af.add_attack_relation("buy_energy_1", "stores_2")
        af.add_attack_relation("buy_energy_1", "stored_1")
        af.add_attack_relation("buy_energy_1", "stored_2")
        af.add_attack_relation("buy_energy_1", "cons_storage")
        af.add_attack_relation("buy_energy_4", "bias_3")

        return af

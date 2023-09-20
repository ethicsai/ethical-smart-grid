import unittest

from smartgrid.rewards.argumentation.lib import (
    AFDM, Argument, JudgingAgent, judgment
)
from smartgrid.rewards.argumentation.lib.exceptions import ArgumentNotFoundError


class TestArgumentation(unittest.TestCase):

    def test_cannot_add_unexisting_attack(self):
        afdm = AFDM()
        self.assertRaises(
            ArgumentNotFoundError,
            lambda: afdm.add_attack_relation('a', 'b')
        )
        # This is raised both for attacker...
        afdm.add_argument(Argument('attacked'))
        self.assertRaises(
            ArgumentNotFoundError,
            lambda: afdm.add_attack_relation('a', 'attacked')
        )
        # ... and attacked arguments
        afdm.add_argument(Argument('attacker'))
        self.assertRaises(
            ArgumentNotFoundError,
            lambda: afdm.add_attack_relation('attacker', 'b')
        )

    def test_simple_grounded(self):
        afdm = AFDM()
        # All 3 arguments are always alive
        a = afdm.add_argument(Argument('a'))
        b = afdm.add_argument(Argument('b'))
        c = afdm.add_argument(Argument('c'))

        afdm.add_attack_relation(attacker='a', attacked='b')
        # We can also add arguments by variable directly
        afdm.add_attack_relation(attacker=b, attacked='c')

        self.assertEqual(len(afdm.A), 3)
        self.assertEqual(len(list(afdm.R)), 2)

        afdm.update_alive_in_grounded_extension()

        self.assertListEqual(afdm.grounded, ['a', 'c'])
        self.assertTrue(a.alive)
        self.assertFalse(b.alive)
        self.assertTrue(c.alive)

    def test_complex(self):
        afdm = AFDM()
        dec = 'moral'
        # A few arguments that have different aliveness condition
        afdm.add_argument(Argument(
            'equity_0.25',
            'Equity greater than 25%',
            lambda s: s['equity'] >= 0.25,
            support=[dec]
        ))
        afdm.add_argument(Argument(
            'equity_0.50',
            'Equity greater than 50%',
            lambda s: s['equity'] >= 0.50,
            support=[dec]
        ))
        afdm.add_argument(Argument(
            'equity_0.75',
            'Equity greater than 75%',
            lambda s: s['equity'] >= 0.75,
            support=[dec]
        ))
        # A few arguments that will be attacked by the equity_n ones
        # (so that we can test whether the attack relationship is deleted)
        afdm.add_argument(Argument(
            'test_0.25',
            'Attacked by equity_0.25'
        ))
        afdm.add_argument(Argument(
            'test_0.50',
            'Attacked by equity_0.50'
        ))
        afdm.add_argument(Argument(
            'test_0.75',
            'Attacked by equity_0.75'
        ))
        # Attacks between `equity_X` and `test_X`
        afdm.add_attack_relation('equity_0.25', 'test_0.25')
        afdm.add_attack_relation('equity_0.50', 'test_0.50')
        afdm.add_attack_relation('equity_0.75', 'test_0.75')
        # The judging agent to simplify the judgment process
        judge = JudgingAgent('equity', afdm, judgment.j_diff)
        # Filter arguments that are alive in this situation
        situation = {'equity': 0.33}
        filtered_afdm = judge._filter_afdm(situation)
        self.assertTrue(filtered_afdm.A['equity_0.25'].alive)
        self.assertFalse(filtered_afdm.A['equity_0.50'].alive)
        self.assertFalse(filtered_afdm.A['equity_0.75'].alive)
        # At this point, the `test_X` arguments should be alive, because
        # their activation function always returns true, and we have not
        # computed the grounded extension yet.
        self.assertTrue(filtered_afdm.A['test_0.25'].alive)
        self.assertTrue(filtered_afdm.A['test_0.50'].alive)
        self.assertTrue(filtered_afdm.A['test_0.75'].alive)
        # Now, compute the grounded extension. Arguments that are attacked by
        # alive arguments (and not defended) will be killed.
        filtered_afdm.update_alive_in_grounded_extension()
        self.assertTrue(filtered_afdm.A['equity_0.25'].alive)
        self.assertFalse(filtered_afdm.A['equity_0.50'].alive)
        self.assertFalse(filtered_afdm.A['equity_0.75'].alive)
        self.assertFalse(filtered_afdm.A['test_0.25'].alive)
        self.assertTrue(filtered_afdm.A['test_0.50'].alive)
        self.assertTrue(filtered_afdm.A['test_0.75'].alive)
        self.assertListEqual(
            filtered_afdm.grounded,
            ['equity_0.25', 'test_0.50', 'test_0.75']
        )
        # Compute the reward and make sure it corresponds to the number
        # of (alive and total) arguments.
        alive_pros = len(filtered_afdm.arguments_supporting(dec))
        total_pros = len(filtered_afdm.arguments_supporting(dec, True))
        alive_cons = len(filtered_afdm.arguments_countering(dec))
        total_cons = len(filtered_afdm.arguments_countering(dec, True))
        self.assertEqual(alive_pros, 1)
        self.assertEqual(total_pros, 3)
        self.assertEqual(alive_cons, 0)
        self.assertEqual(total_cons, 0)
        reward = judgment.j_diff(filtered_afdm, dec)
        self.assertEqual(reward, 1 / 3)


if __name__ == '__main__':
    unittest.main()

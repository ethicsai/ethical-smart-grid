import unittest

from smartgrid import World
from smartgrid.agents import DataOpenEIConversion, Agent, Action
from smartgrid.agents.profile import comfort, ProductionProfile
from smartgrid.make_env import find_profile_data
from smartgrid.util import RandomEnergyGenerator


class TestAction(unittest.TestCase):

    def _create_world(self):
        # Prepare the World
        converter = DataOpenEIConversion()
        profile = converter.load(
            'Household',
            find_profile_data('openei', 'profile_residential_annually.npz'),
            comfort.flexible_comfort_profile
        )
        # Disable production to simplify testing.
        profile.production_fn = ProductionProfile([0])
        agent = Agent('Household1', profile)
        generator = RandomEnergyGenerator()
        world = World([agent], generator)
        world.reset()
        return world

    def test_store_more_than_capacity(self):
        """
        Test that we can ask to store more than the capacity; the battery
        should be filled as much as possible, and the overflow should be
        shifted to the storage consumption.
        """
        world = self._create_world()
        agent = world.agents[0]

        # This should already be the value, but let us set it explicitly.
        agent.profile.max_storage = 500
        # Reset the storage (no production).
        agent.state.storage = 0
        # Fake an action and test that it works correctly.
        # We store more energy than the `max_storage` allows; the storage should
        # not overflow.
        store = 1000
        action = Action(grid_consumption=0,
                        storage_consumption=0,
                        store_energy=store,
                        give_energy=0,
                        buy_energy=0,
                        sell_energy=0)
        agent.intended_action = action
        world.step()

        # The storage should be equal to the max storage
        self.assertEqual(agent.state.storage, agent.profile.max_storage)

        # All action parameters' should be at least 0
        for value in agent.enacted_action._asdict().values():
            self.assertGreaterEqual(value, 0)

        # The overflow should be shifted to the storage consumption
        overflow = store - agent.profile.max_storage
        self.assertEqual(agent.enacted_action.storage_consumption, overflow)

    def test_buy_more_than_capacity(self):
        """
        Test that we can ask to buy more than the capacity; the battery should
        be filled as much as possible, and the overflow should be shifted to
        the storage consumption.
        """
        world = self._create_world()
        agent = world.agents[0]

        # This should already be the value, but let us set it explicitly.
        agent.profile.max_storage = 500
        # Reset the storage (no production).
        agent.state.storage = 0
        # Fake an action and test that it works correctly.
        # We buy more energy than the `max_storage` allows; the storage should
        # not overflow.
        buy = 1000
        action = Action(grid_consumption=0,
                        storage_consumption=0,
                        store_energy=0,
                        give_energy=0,
                        buy_energy=buy,
                        sell_energy=0)
        agent.intended_action = action
        world.step()

        # The storage should be equal to the max storage
        self.assertEqual(agent.state.storage, agent.profile.max_storage)

        # All action parameters' should be at least 0
        for value in agent.enacted_action._asdict().values():
            self.assertGreaterEqual(value, 0)

        # The overflow should be shifted to the storage consumption
        overflow = buy - agent.profile.max_storage
        self.assertEqual(agent.enacted_action.storage_consumption, overflow)

    def test_consume_storage_more_than_battery(self):
        """
        Test that we can ask to consume more than the available battery; the
        consumption should be as much as possible (but not higher than the
        battery), and the battery should be emptied.
        """
        world = self._create_world()
        agent = world.agents[0]

        # This should already be the value, but let us set it explicitly.
        agent.profile.max_storage = 500
        # Reset the storage to an arbitrary value, e.g., 350.
        agent.state.storage = 350
        # Fake an action and test that it works correctly.
        # We consume more energy than the `storage` allows; the storage should
        # be emptied, and the consumption reduced.
        consume = 400
        action = Action(grid_consumption=0,
                        storage_consumption=consume,
                        store_energy=0,
                        give_energy=0,
                        buy_energy=0,
                        sell_energy=0)
        agent.intended_action = action
        world.step()

        # The storage should be emptied.
        self.assertEqual(agent.state.storage, 0)

        # All action parameters' should be at least 0
        for value in agent.enacted_action._asdict().values():
            self.assertGreaterEqual(value, 0)

        # The consumption should be less than desired.
        self.assertEqual(agent.enacted_action.storage_consumption, 350)

    def test_give_storage_more_than_battery(self):
        """
        Test that we can ask to give more than the available battery; the
        battery should be emptied, and the given energy should be reduced.
        """
        world = self._create_world()
        agent = world.agents[0]

        # This should already be the value, but let us set it explicitly.
        agent.profile.max_storage = 500
        # Reset the storage to an arbitrary value, e.g., 350.
        agent.state.storage = 350
        # Fake an action and test that it works correctly.
        # We give more energy than the `storage` allows; the storage should
        # be emptied, and the `given` reduced.
        give = 400
        action = Action(grid_consumption=0,
                        storage_consumption=0,
                        store_energy=0,
                        give_energy=give,
                        buy_energy=0,
                        sell_energy=0)
        agent.intended_action = action
        world.step()

        # The storage should be emptied.
        self.assertEqual(agent.state.storage, 0)

        # All action parameters' should be at least 0
        for value in agent.enacted_action._asdict().values():
            self.assertGreaterEqual(value, 0)

        # The given energy should be equal to the previous storage (350).
        self.assertEqual(agent.enacted_action.give_energy, 350)

    def test_sell_storage_more_than_battery(self):
        """
        Test that we can ask to sell more than the available battery; the
        storage should be emptied, and the sold energy be reduced.
        """
        world = self._create_world()
        agent = world.agents[0]

        # This should already be the value, but let us set it explicitly.
        agent.profile.max_storage = 500
        # Reset the storage to an arbitrary value, e.g., 350.
        agent.state.storage = 350
        # Fake an action and test that it works correctly.
        # We sell more energy than the `storage` allows; the storage should
        # be emptied, and the sold energy be reduced.
        sell = 400
        action = Action(grid_consumption=0,
                        storage_consumption=0,
                        store_energy=0,
                        give_energy=0,
                        buy_energy=0,
                        sell_energy=sell)
        agent.intended_action = action
        world.step()

        # The storage should be emptied.
        self.assertEqual(agent.state.storage, 0)

        # All action parameters' should be at least 0
        for value in agent.enacted_action._asdict().values():
            self.assertGreaterEqual(value, 0)

        # The sold energy should be equal to the previous storage (350).
        self.assertEqual(agent.enacted_action.sell_energy, 350)

    def test_buy_and_give(self):
        """
        Test that we can give the energy bought at the same step.
        """
        world = self._create_world()
        agent = world.agents[0]

        # This should already be the value, but let us set it explicitly.
        agent.profile.max_storage = 500
        # Reset the storage (no production).
        agent.state.storage = 0
        # Fake an action and test that it works correctly.
        # We want to give energy, we currently have 0, but we buy energy at the
        # same time.
        buy = 300
        give = 250
        action = Action(grid_consumption=0,
                        storage_consumption=0,
                        store_energy=0,
                        give_energy=give,
                        buy_energy=buy,
                        sell_energy=0)
        agent.intended_action = action
        world.step()

        # The storage should be what remains of the bought energy after giving.
        self.assertEqual(agent.state.storage, buy - give)

        # All action parameters' should be at least 0
        for value in agent.enacted_action._asdict().values():
            self.assertGreaterEqual(value, 0)

        # The bought energy should be as requested.
        self.assertEqual(agent.enacted_action.buy_energy, buy)
        # The given energy should be as requested.
        self.assertEqual(agent.enacted_action.give_energy, give)


if __name__ == '__main__':
    unittest.main()

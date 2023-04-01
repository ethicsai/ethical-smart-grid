import unittest

import numpy as np

from smartgrid import World
from smartgrid.agents import Agent, AgentProfile, Action
from smartgrid.agents.profile import NeedProfile, ProductionProfile
from smartgrid.util import RandomEnergyGenerator


def simple_comfort(consumption: float, need: float) -> float:
    # Simple linear function
    # (`10e-300` is used to avoid division by 0)
    comfort = consumption / (need + 10e-300)
    # Restrict the result to [0,1]
    comfort = np.clip(comfort, 0.0, 1.0)
    return float(comfort)


class TestWorld(unittest.TestCase):
    """
    Test that the World (physical simulation) works correctly.
    """

    def test_world(self):
        # Hardcode profiles (instead of loading them with DataConversion)
        needs = [100, 200, 300, 200]
        productions = [50, 20, 35, 10]
        profile_household = AgentProfile(
            name='Household',
            need_profile=NeedProfile(needs),
            production_profile=ProductionProfile(productions),
            max_storage=400,
            action_space_low=np.array([0, 0, 0, 0, 0, 0]),
            action_space_high=np.array([750, 750, 750, 750, 750, 750]),
            action_dim=6,
            comfort_fn=simple_comfort
        )

        # Create Agents
        agent1 = Agent('Household1', profile_household)
        agent2 = Agent('Household2', profile_household)

        # Create EnergyGenerator
        energy_generator = RandomEnergyGenerator(
            lower_proportion=0.8,
            upper_proportion=1.2
        )

        # Create World
        world = World(
            agents=[agent1, agent2],
            energy_generator=energy_generator
        )

        # Initialize the World by resetting it
        world.reset()
        # Test that current/min/max need are coherent
        current_need = world.current_need
        self.assertGreaterEqual(current_need, world.min_needed_energy)
        self.assertLessEqual(current_need, world.max_needed_energy)
        # Test that the agent's need, production and storage are correct
        self.assertEqual(agent1.need, needs[0])
        self.assertEqual(agent2.need, needs[0])
        self.assertEqual(agent1.production, productions[0])
        self.assertEqual(agent2.production, productions[0])
        self.assertEqual(agent1.state.storage, productions[0])
        self.assertEqual(agent2.state.storage, productions[0])
        # Manually set agents' actions
        agent1.intended_action = Action(*agent1.profile.action_space.sample())
        agent2.intended_action = Action(*agent2.profile.action_space.sample())

        for step in range(10):
            with self.subTest(step=step):
                # We need to memorize the agents' needs (because `world.step()`
                # will update them).
                agent1_need = agent1.state.need
                agent2_need = agent2.state.need
                # Update the world to get to the next time step
                world.step()
                # Test again the current/min/max need
                current_need = world.current_need
                self.assertGreaterEqual(current_need, world.min_needed_energy)
                self.assertLessEqual(current_need, world.max_needed_energy)
                # Test that agent's comfort is coherent
                self.assertAlmostEqual(agent1.comfort, simple_comfort(
                    agent1.enacted_action.grid_consumption + agent1.enacted_action.storage_consumption,
                    agent1_need
                ))
                self.assertAlmostEqual(agent2.comfort, simple_comfort(
                    agent2.enacted_action.grid_consumption + agent2.enacted_action.storage_consumption,
                    agent2_need
                ))


if __name__ == '__main__':
    unittest.main()

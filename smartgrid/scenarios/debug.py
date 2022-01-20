import random

from smartgrid.scenarios._load_agent import create_household
from smartgrid.world import World
from smartgrid.util import RandomEnergyGenerator


class Scenario(object):

    def make_world(self):
        energy_generator = RandomEnergyGenerator()
        world = World(energy_generator)
        # Create the Households agents
        for i in range(2):
            agent = create_household(i)
            world.agents.append(agent)
        return world

    def reset_world(self, world):
        return world

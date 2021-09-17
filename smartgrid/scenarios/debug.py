import random

from smartgrid.scenarios._load_agent import create_household
from smartgrid.world import World


def compute_energy(env):
    return random.randint(0, 1000)


class Scenario(object):

    def make_world(self):
        world = World(compute_energy)
        # Create the Households agents
        for i in range(2):
            agent = create_household(i)
            world.agents.append(agent)
        return world

    def reset_world(self, world):
        return world

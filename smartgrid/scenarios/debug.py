import numpy as np
import random

from gym import spaces

from smartgrid.agent import Agent
from smartgrid.util import comforts
from smartgrid.world import World


def compute_energy(env):
    return random.randint(0, 1000)

def compute_production(step):
    return random.randint(0, 100)

def compute_need(step):
    return random.randint(0, 800)


def create_household(i):
    action_range = spaces.Box(
        low=np.array([0, 0, 0, 0, 0, 0]),
        high=np.array([2000, 2000, 2000, 2000, 2000, 2000]),
        dtype=np.int
    )
    max_battery = 500
    comfort_fn = comforts.comfort_flexible
    name = f'Household[Flexible] {i}'
    agent = Agent(name,
                  action_range,
                  max_battery,
                  comfort_fn,
                  compute_need,
                  compute_production
                  )
    return agent


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

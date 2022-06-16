import gym
import numpy as np
from gym.vector.utils import spaces

from smartgrid.agents.agent import Action
from smartgrid.world import World


class SmartGrid(gym.Env):
    """
    The Smart Grid multi-agent environment.
    """

    metadata = {
        'render.modes': ['text'],
    }

    # reward_range = (0.0, +1.0)

    def __init__(self, world: World):
        self.world = world

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.world.agents:
            self.action_space.append(agent.profile.action_space)
            self.observation_space.append(world.observation.get_observation_space())

        self.action_space = np.array(self.action_space)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = [False] * len(self.world.agents)

        # Reload list of agents
        # self.agents = self.world.policy_agents

        # Set action for each agent (will be performed in `world.step()`)
        for i, agent in enumerate(self.world.agents):
            agent.intended_action = Action(*(action_n[i]))

        # Next step of simulation
        self.world.step()

        # Compute next observations and rewards
        for agent in self.world.agents:
            # TODO move that to Environment or a wrapper
            obs_n.append(self.world.get_observation_agent(agent))
            reward_n.append(self.world.get_reward(agent))

        obs = {
            "global": self.world.get_observation_global(),
            "local": [self.world.get_observation_agent(agent) for agent in self.world.agents]
        }

        # Only used for visualization, performance metrics, ...
        info_n = self.world.get_info(reward_n)
        return obs, reward_n, done_n, info_n

    def reset(self):
        self.world.reset()
        obs = {
            "global": self.world.get_observation_global(),
            "local": [self.world.get_observation_agent(agent) for agent in self.world.agents]
        }
        return obs

    def render(self, mode='text'):
        pass

    @property
    def n_agent(self):
        return len(self.world.agents)

    @property
    def observation_shape(self):
        return self.world.observation_shape


    def observation_space_per_agent(self, agent_num: int):
        all_space = self.world.observation_space
        to_return = all_space['global'].spaces
        to_return.update(all_space['local'][str(agent_num)].spaces)

        return spaces.Dict(to_return)

    @property
    def world_observation_space(self):
        return self.world.observation_space

    def observation_space_per_agent(self, agent_num: int):
        all_space = self.world.observation_space
        to_return = all_space['global'].spaces
        to_return.update(all_space['local'][str(agent_num)].spaces)

        return spaces.Dict(to_return)

    @property
    def agents(self):
        return self.world.agents

    @property
    def available_energy(self):
        return self.world.available_energy

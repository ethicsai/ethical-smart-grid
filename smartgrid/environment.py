import numpy as np
import gym
from gym.spaces import Box

from smartgrid.agents.agent import Action
from smartgrid.observations import Observation
from smartgrid.rewards import equity_reward


class SmartGrid(gym.Env):
    """
    The Smart Grid multi-agent environment.
    """

    metadata = {
        'render.modes': ['text'],
    }
    # reward_range = (0.0, +1.0)

    def __init__(self, world):
        self.world = world
        self.agents = world.agents
        self.n_agents = len(self.agents)

        # Scenario callbacks
        # self.reset_callback = None
        self.reward_callback = equity_reward
        # self.info_callback = None
        # self.done_callback = None

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            self.action_space.append(agent.action_space)
            obs_space = Box(low=0.0, high=1.0, shape=(len(Observation._fields),))
            self.observation_space.append(obs_space)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = [False] * self.n_agents
        info_n = {}

        # Reload list of agents
        # self.agents = self.world.policy_agents

        # Set action for each agent (will be performed in `world.step()`)
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)

        # Next step of simulation
        self.world.step()

        # Compute next observations and rewards
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))

        # Only used for visualization, performance metrics, ...
        mean_reward = np.mean(reward_n)
        info_n['global_reward'] = mean_reward

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.world.reset()
        self.agents = self.world.agents
        self.n_agents = len(self.agents)
        obs_n = [
            self._get_obs(agent) for agent in self.agents
        ]
        return obs_n

    def render(self, mode='text'):
        pass

    def _get_obs(self, agent):
        return Observation.compute(self, agent)

    def _get_reward(self, agent):
        return self.reward_callback(self, agent)

    def _set_action(self, action, agent):
        agent.action = Action(*action)

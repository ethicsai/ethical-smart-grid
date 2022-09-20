import gym
import numpy as np
from gym.vector.utils import spaces

from smartgrid.agents.agent import Action
from smartgrid.world import World


class SmartGrid(gym.Env):
    """
    The Smart Grid multi-agent environment is the jointure between :py:class:`Env` of Gym libraries and the simulator.
    The Simulator work like an environment Gym, please note that the key feature is multi-agent handling physically \
    inside the environment. Key methods for advancing and resetting are :py:meth:`step` and :py:meth:`reset`.

    Contains :py:attr:`world`, an instance of :py:class:`World`, a class defining how to handle the physical component \
    of a SmartGrid, like energy flow in the network and the inner flow of entity by the :py:class:`Agent`.
    """

    metadata = {
        'render.modes': ['text'],
    }

    # reward_range = (0.0, +1.0)

    def __init__(self, world: World):
        """
        Initialization of the Smartgrid. It constructs gym attributes :py:attr:`action_space` and \
        :py:attr:`observation_space` for generalize all agent attributes.

        :param world: the physical :py:class:`World` of the Smart Grid, it is constructed before with multiple field \
        (refers to :py:class:`Scenario` class). the instance is handled by the Smart Grid for standardisation (by Gym).
        """
        self.world = world

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.world.agents:
            self.action_space.append(agent.profile.action_space)
            self.observation_space.append(world.observation.get_observation_space())

        self.action_space = np.array(self.action_space)

    def step(self, action_n):
        """
        Methods that compute the next phase of the simulation.
        :param action_n: all action choose by external intelligence (need a corresponding amount with number of agent).
        :return: All information for deciding and learning. Returns in total four field:
            - obs: a dict that contains the 'global' observation of the network and all 'local' information \
            of physical agent.
            - reward_n: the list of reward of an intelligent agent decision (can be multiple).
            - done_n: not really used in the SmartGrid, can be expanded for future need of prevent failure.
            - info_n: information concerning all agent. Can be expanded by the :py:class:`Agent`.
        """
        obs_n = []
        reward_n = []
        done_n = [False] * len(self.world.agents)

        # Set action for each agent (will be performed in `world.step()`)
        for i, agent in enumerate(self.world.agents):
            agent.intended_action = Action(*(action_n[i]))

        # Next step of simulation
        self.world.step()

        # Get next observations and rewards
        for agent in self.world.agents:
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
        """
        Reset the SmartGrid, all intern class call his reset method. Like that all components will reset.
        After that, the observation is construct.
        :return: the first observation after resetting.
        """
        self.world.reset()
        obs = {
            "global": self.world.get_observation_global(),
            "local": [self.world.get_observation_agent(agent) for agent in self.world.agents]
        }
        return obs

    def render(self, mode='text'):
        """
        No render have been configured for now. Metrics can be observed in mathematical way by the object return by \
        :py:meth:`step`.
        :param mode: not used
        :return: None
        """
        pass

    @property
    def n_agent(self):
        """
        Property that reduce indirection.
        :return: number of instance of :py:class:`Agent` in :py:class:`World`.
        """
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
    def agents(self):
        """
        Property that reduce indirection.
        :return: instance of :py:class:`Agent` in :py:class:`World`.
        """
        return self.world.agents

    @property
    def available_energy(self):
        """
        Property that reduce indirection.
        :return: available energy value already compute during :py:meth:`step`.
        """
        return self.world.available_energy

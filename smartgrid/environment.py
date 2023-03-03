import gymnasium
import numpy as np

from smartgrid.agents import Action
from smartgrid.rewards import RewardCollection
from smartgrid.world import World
from smartgrid.observation import ObservationManager


class SmartGrid(gymnasium.Env):
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

    def __init__(self,
                 world: World,
                 obs_manager: ObservationManager,
                 rewards):
        """
        Initialization of the Smartgrid. It constructs gym attributes :py:attr:`action_space` and \
        :py:attr:`observation_space` for generalize all agent attributes.

        :param world: the physical :py:class:`World` of the Smart Grid, it is constructed before with multiple field \
        (refers to :py:class:`Scenario` class). the instance is handled by the Smart Grid for standardisation (by Gym).
        """
        self.world = world
        self.observation_manager = obs_manager
        self.reward_calculator = RewardCollection(rewards)

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        obs_space = self.observation_manager.observation.get_observation_space()
        for agent in self.world.agents:
            self.action_space.append(agent.profile.action_space)
            self.observation_space.append(obs_space)

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
        done_n = [False] * len(self.world.agents)

        # Set action for each agent (will be performed in `world.step()`)
        for i, agent in enumerate(self.world.agents):
            agent.intended_action = Action(*(action_n[i]))

        # Next step of simulation
        self.world.step()

        # Get next observations and rewards
        obs = self._get_obs()
        reward_n = self._get_reward()

        # Only used for visualization, performance metrics, ...
        info_n = self._get_info(reward_n)

        return obs, reward_n, done_n, info_n

    def reset(self, seed=None, options=None):
        """
        Reset the SmartGrid to its initial state.

        This method will call the `reset` method on the internal objects,
        e.g., the :class:`World`, the :class:`Agent`\\ s, etc.
        Despite its name, it **must** be used first and foremost to get the
        initial observations.

        :param seed: An optional seed (int) to configure the random generators
            and ensure reproducibility.

        :param options: An optional dictionary of arguments to further
            configure the simulator. Currently unused.

        :return: The first (initial) observations for each agent in the World.
        """
        super().reset(seed=seed)
        self.world.reset()

        obs = self._get_obs()
        return obs

    def render(self, mode='text'):
        """
        Render the current state of the simulator to the screen.

        .. note:: No render have been configured for now.
            Metrics' values can be observed directly through the object
            returned by :py:meth:`step`.

        :param mode: Not used

        :return: None
        """
        pass

    def _get_obs(self):
        """
        Determine the observations for all agents.

        .. note:: As a large part of the observations are shared ("global"),
            we use instead of the traditional list (1 obs per agent) a dict,
            containing:
            - `global` the global observations, shared by all agents;
            - `local` a list of local observations, one item for each agent.

        :return: A dictionary containing `global` and `local`.
        """
        return {
            "global": self.observation_manager.compute_global(self.world),
            "local": [
                self.observation_manager.compute_agent(self.world, agent)
                for agent in self.world.agents
            ]
        }

    def _get_reward(self):
        """
        Determine the reward for each agent.

        Rewards describe to which degree the agent's action was appropriate,
        w.r.t. moral values. These moral values are encoded in the reward
        function(s), see :py:mod:`smartgrid.rewards` for more details on them.

        Reward functions may comprise multiple objectives. In such cases, they
        can be aggregated so that the result is a single float (which is used
        by most of the decision algorithms).
        This behaviour (whether to aggregate, and how to aggregate) is
        controlled by the :py:attr:`.reward_calculator`, see
        :py:class:`.RewardCollection` for details.

        :return: A list of rewards, one element per agent. The element itself
            is a dict which contains at least one reward, indexed by the
            reward's name.
        """
        return [
            self.reward_calculator.compute(self.world, agent)
            for agent in self.world.agents
        ]

    def _get_info(self, reward_n):
        """
        Return additional information on the world (for the current time step).

        Information contain the rewards, for each agent.

        :param reward_n: The list of rewards, one for each agent.

        :return: A dict, containing an element with key ``rewards``.
            This element is itself a dict, indexed by the agents' names, and
            whose value is their reward.
        """
        info_n = {"rewards": {}}

        for i, agent in enumerate(self.agents):
            info_n["rewards"][agent.name] = reward_n[i]

        return info_n

    @property
    def n_agent(self):
        """
        Property that reduce indirection.
        :return: number of instance of :py:class:`Agent` in :py:class:`World`.
        """
        return len(self.world.agents)

    @property
    def observation_shape(self):
        return self.observation_manager.shape

    @property
    def agents(self):
        """
        Property that reduce indirection.
        :return: instance of :py:class:`Agent` in :py:class:`World`.
        """
        return self.world.agents

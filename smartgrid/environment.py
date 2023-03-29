"""
The SmartGrid environment is the main entrypoint.
"""

import warnings

import gymnasium
from gymnasium import Space

from smartgrid.agents import Action
from smartgrid.rewards import RewardCollection
from smartgrid.world import World
from smartgrid.observation import ObservationManager

from typing import List, Optional


class SmartGrid(gymnasium.Env):
    """
    The SmartGrid environment is the main entrypoint.

    It simulates a smart grid containing multiple agents (prosumers: producers
    and consumers) who must learn to distribute and exchange energy between
    them, to satisfy their comfort while taking into account various ethical
    considerations.

    This class extends the standard :py:class:`gym.Env <gymnasium.core.Env>`
    in order to be easily used with different learning algorithms.
    However, a key feature of this environment is that multiple agents co-exist,
    hence some changes have been made to the standard Gym API.
    Notably: the :py:attr:`~smartgrid.environment.SmartGrid.action_space` and
    :py:attr:`~smartgrid.environment.SmartGrid.observation_space` are lists of
    :py:class:`~gymnasium.spaces.Space` instead of just a Space; the
    :py:meth:`~smartgrid.environment.SmartGrid.step` method returns list and
    dicts instead of single elements.
    """

    metadata = {
        'render.modes': ['text'],
    }

    action_space: List[Space]
    """
    The list of action spaces for all Agents.
    """

    observation_space: List[Space]
    """
    The list of observation spaces for all Agents.
    
    Because the observation space is in practice split between *global* and
    *local* observations, this might not exactly correspond, please see
    the :py:meth:`~smartgrid.environment.SmartGrid._get_obs` for details.
    """

    observation_manager: ObservationManager
    """
    The observation manager, responsible for creating observations each step.
    
    Can be configured (extended) to return different observations.
    """

    max_step: Optional[int]
    """
    The maximum number of steps allowed in the environment (or None by default).
    
    As the environment is not episodic, it does not have a way to terminate
    (i.e., agents cannot "solve" their task nor "die"). The maximum number
    of steps is a way to limit the simulation and force the environment to
    terminate. In practice, it simply determines the ``truncated`` return value
    of :py:meth:`~smartgrid.environment.SmartGrid.step`. This return value, in
    turn, acts as a signal for the external *interaction loop*.
    By default, or when sent to ``None``, ``truncated`` will always return
    ``false``, which means that the environment can be used forever.
    """

    reward_calculator: RewardCollection
    """
    The RewardCollection, responsible for determining agents' rewards each step.
    
    This environment has a (partial) support for *multi-objective* use-cases,
    i.e., multiple reward functions can be used at the same time. The
    :py:class:`~smartgrid.rewards.reward_collection.RewardCollection` is used
    to hold all these functions, and compute the rewards for all functions, and
    for all agents, at each time step. It returns a list of dicts (multiple
    rewards for each agent), which can be scalarized to a list of floats
    (single reward for each agent) by using a wrapper over this environment.
    See the :py:mod:`~smartgrid.wrappers.reward_aggregator` module for details.
    """

    world: World
    """
    The simulated world in which the SmartGrid exists.
    
    The world is responsible for handling all agents and "physical" interactions
    between the smart grid elements.
    """

    # reward_range = (0.0, +1.0)

    def __init__(self,
                 world: World,
                 rewards,
                 max_step=None,
                 obs_manager: ObservationManager = None):
        """
        Create the SmartGrid environment.

        This sets most attributes of the environment, including the
        :py:attr:`~smartgrid.environment.SmartGrid.action_space` and
        :py:attr:`~smartgrid.environment.SmartGrid.observation_space`.

        .. warning::
            Remember that the env is not usable until you call :py:meth:`.reset` !

        :param world: The "physical" :py:class:`.World` of the Smart Grid
            in which the simulation happens. The world contains the agents,
            the energy generator, and handles the agents' actions.

        :param rewards: The list of reward functions that should be used.
            Usually, a list of a single element (for single-objective RL),
            but multiple reward functions can be used.

        :param max_step: The maximal number of steps allowed in the environment.
            By default, the environment never terminates on its own: the
            interaction loop must be stopped from the outside. If this value
            is set, the :py:meth:`.step` method will return ``truncated=True``
            when ``max_step`` steps have been done. Subsequent calls will raise
            a warning.

        :param obs_manager: (Optional) The :py:class:`.ObservationManager` that
            will be used to determine :py:class:`.Observation`\\ s at each
            time step. This parameter can be used to extend this process, and
            generate different observations. It can (and will in most cases)
            be left to its default value.

        :return: An instance of SmartGrid.
        """
        self.world = world
        self.max_step = max_step
        if obs_manager is None:
            obs_manager = ObservationManager()
        self.observation_manager = obs_manager
        self.reward_calculator = RewardCollection(rewards)

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        obs_space = self.observation_manager.observation.get_observation_space()
        for agent in self.world.agents:
            self.action_space.append(agent.profile.action_space)
            self.observation_space.append(obs_space)

    def step(self, action_n):
        """
        Advance the simulation to the next step.

        This method takes the actions' decided by agents (learning algorithms),
        and sends them to the :py:class:`.World` so it can update itself based
        on these actions.
        Then, the method computes the new observations and rewards, and returns
        them so that agents can decide the next action.

        :param action_n: The list of actions (vectors of parameters that must
            be coherent with the agent's action space), one action for each
            agent.

        :return: A tuple containing information about the next (new) state:

            - ``obs_n``: A dict that contains the observations about the next
              state, please see :py:meth:`._get_obs` for details about the
              dict contents.
            - ``reward_n``: A list containing the rewards for each agent,
              please see :py:meth:`._get_reward` for details about its content.
            - ``terminated_n``: A list of boolean values indicating, for each
              agent, whether the agent is "terminated", e.g., completed its
              task or failed. Currently, always set to ``False``: agents
              cannot complete nor fail (this is not an episodic environment).
            - ``truncated_n``: A list of boolean values indicating, for each
              agent, whether the agent should stop acting, because, e.g., the
              environment has run out of time. See :py:attr:`.max_step` for
              details.
            - ``info_n``: A dict containing additional information about the
              next state, please see :py:meth:`._get_info` for details about
              its content.

        .. note: ``terminated_n`` and ``truncated_n`` replace the previous
            (pre-Gym-v26) ``done_n`` return value. The ``done`` value
            can be obtained with ``all(terminated_n) or all(truncated_n)``.
        """
        if self.max_step is not None and self.world.current_step >= self.max_step:
            warnings.warn(f'max_step was set to {self.max_step}, but step'
                          f'{self.world.current_step} was requested.')

        # Set action for each agent (will be performed in `world.step()`)
        for i, agent in enumerate(self.world.agents):
            agent.intended_action = Action(*(action_n[i]))

        # Next step of simulation
        self.world.step()

        # Get next observations and rewards
        obs = self._get_obs()
        reward_n = self._get_reward()

        n_agent = self.n_agent
        terminated_n = [False] * n_agent
        if self.max_step is None:
            truncated_n = [False] * n_agent
        else:
            # We use `-1` because the first step is the `0th`.
            truncated_n = [self.world.current_step >= self.max_step - 1] * n_agent

        # Only used for visualization, performance metrics, ...
        info_n = self._get_info(reward_n)

        return obs, reward_n, terminated_n, truncated_n, info_n

    def reset(self, seed=None, options=None):
        """
        Reset the SmartGrid to its initial state.

        This method will call the ``reset`` method on the internal objects,
        e.g., the :py:class:`.World`, the :py:class:`.Agent`\\ s, etc.
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
        self.reward_calculator.reset()

        obs = self._get_obs()
        return obs

    def render(self, mode='text'):
        """
        Render the current state of the simulator to the screen.

        .. note:: No render have been configured for now.
            Metrics' values can be observed directly through the object
            returned by :py:meth:`.step`.

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

            - ``global`` the global observations, shared by all agents;
            - ``local`` a list of local observations, one item for each agent.

        :return: A dictionary containing ``global`` and ``local``.
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
        """Number of agents contained in the environment (world)."""
        return len(self.world.agents)

    @property
    def observation_shape(self):
        """The shape, i.e., number of dimensions, of the observation space."""
        return self.observation_manager.shape

    @property
    def agents(self):
        """The list of agents contained in the environment (world)."""
        return self.world.agents

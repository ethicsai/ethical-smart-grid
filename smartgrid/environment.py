"""
The SmartGrid environment is the main entrypoint.
"""
import warnings
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
from gymnasium import Space
import pettingzoo
# Import a few type hints to make the documentation clearer
from pettingzoo.utils.env import AgentID, ObsDict, ActionDict

from smartgrid.agents import Action, Agent
from smartgrid.rewards import RewardCollection, Reward
from smartgrid.world import World
from smartgrid.observation import ObservationManager

# A few additional type hints to help (similar to PettingZoo's)
InfoType = Dict[str, Any]
InfoDict = Dict[AgentID, InfoType]
RewardsType = Dict[str, float]
RewardsDict = Dict[AgentID, RewardsType]


class SmartGrid(pettingzoo.ParallelEnv):
    """
    The SmartGrid environment is the main entrypoint.

    It simulates a smart grid containing multiple agents (prosumers: producers
    and consumers) who must learn to distribute and exchange energy between
    them, to satisfy their comfort while taking into account various ethical
    considerations.

    This class extends the standard :py:class:`pettingzoo.utils.ParallelEnv`
    in order to be easily used with different learning algorithms.
    This is a multi-agent version of the well-known Gym API.
    """

    metadata = {
        'render.modes': ['text'],
    }

    _np_random: np.random.Generator
    """
    The pseudo-random number generator (PRNG), for reproducibility.

    It should usually not be accessed by the user, and must be passed down to
    elements of the SmartGrid (e.g., :py:class:`~.World`) that need it.
    The generator is set by the :py:meth:`~.reset` method, optionally with a
    specific seed.
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
                 rewards: List[Reward],
                 max_step: int = None,
                 obs_manager: ObservationManager = None):
        """
        Create the SmartGrid environment.

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
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.world.agents:
            self.observation_spaces[agent.name] = obs_manager.observation.space(
                self.world,
                agent
            )
            self.action_spaces[agent.name] = agent.profile.action_space

    def step(
        self,
        actions: ActionDict
    ) -> Tuple[
        ObsDict, RewardsDict, Dict[AgentID, bool], Dict[AgentID, bool], InfoDict
    ]:
        """
        Advance the simulation to the next step.

        This method takes the actions' decided by agents (learning algorithms),
        and sends them to the :py:class:`.World` so it can update itself based
        on these actions.
        Then, the method computes the new observations and rewards, and returns
        them so that agents can decide the next action.

        :param actions: The dictionary of actions, indexed by the agent's name,
            where each action is a vector of parameters that must be coherent
            with the agent's action space.

        :return: A tuple containing information about the next (new) state:

            - ``obs_n``: A dict that contains the observations about the next
              state; please see :py:meth:`._get_obs` for details about the
              dict contents.
            - ``reward_n``: A dict containing the rewards for each agent;
              please see :py:meth:`._get_reward` for details about its content.
            - ``terminated_n``: A dict of boolean values indicating, for each
              agent, whether the agent is "terminated", e.g., completed its
              task or failed. Currently, always set to ``False``: agents
              cannot complete nor fail (this is not an episodic environment).
            - ``truncated_n``: A dict of boolean values indicating, for each
              agent, whether the agent should stop acting, because, e.g., the
              environment has run out of time. See :py:attr:`.max_step` for
              details.
            - ``info_n``: A dict containing additional information about the
              next state, please see :py:meth:`._get_info` for details about
              its content.

        .. note: ``terminated_n`` and ``truncated_n`` replace the previous
            (pre-Gym-v26) ``done_n`` return value. The ``done`` value
            can be obtained with
            ``all(terminated_n.values()) or all(truncated_n.values())``.
        """
        if self.max_step is not None and self.world.current_step >= self.max_step:
            warnings.warn(f'max_step was set to {self.max_step}, but step'
                          f'{self.world.current_step} was requested.')

        # Set action for each agent (will be performed in `world.step()`)
        for agent_name, action in actions.items():
            agent = self.world.agents_by_name.get(agent_name)
            assert agent is not None, f'Agent {agent_name} not found'
            agent.intended_action = Action(*action)

        # Next step of simulation
        self.world.step()

        # Get next observations and rewards
        obs = self._get_obs()
        reward_n = self._get_reward()

        # Agents are never "terminated" (they cannot die or stop acting)
        terminated_n = {agent.name: False for agent in self.world.agents}

        # Agents are truncated only if the `max_step` is defined, and higher
        # than the current time step. They are either all truncated, or none
        # of them is.
        if self.max_step is None:
            truncated_n = {agent.name: False for agent in self.world.agents}
        else:
            # We use `-1` because the first step is the `0th`.
            truncated = self.world.current_step >= self.max_step - 1
            truncated_n = {agent.name: truncated for agent in self.world.agents}

        # Only used for visualization, performance metrics, ...
        info_n = self._get_info(reward_n)

        return obs, reward_n, terminated_n, truncated_n, info_n

    def reset(
        self,
        seed: Optional[int] = None,
        options: Dict = None
    ) -> Tuple[ObsDict, InfoDict]:
        """
        Reset the SmartGrid to its initial state.

        This method will call the ``reset`` method on the internal objects,
        e.g., the :py:class:`.World`, the :py:class:`.Agent`\\ s, etc.
        Despite its name, it **must** be used first and foremost to get the
        initial observations.

        :param seed: An optional seed (int) to configure the random generators
            and ensure reproducibility.
            Note: this does **not** change the global generators (Python
            `random` and NumPy `np.random`). SmartGrid components must rely
            on the :py:attr:`~SmartGrid._np_random` attribute.

        :param options: An optional dictionary of arguments to further
            configure the simulator. Currently unused.

        :return: A tuple containing the observations and additional information
            for the first (initial) time step, in this order. There is no
            additional information in the current version, but an empty dict is
            still returned to be coherent with the base API.
            The observations is a dictionary indexed by agents' name,
            containing their initial observations, for each agent in the
            :py:class:`~smartgrid.world.World`.
        """
        self._np_random = np.random.default_rng(seed)
        self.observation_manager.reset()
        self.world.reset(self._np_random)
        self.reward_calculator.reset()

        obs = self._get_obs()
        # PettingZoo requires the infos to contain a dictionary for each agent,
        # even if the dictionary itself is empty.
        infos = {agent_name: {} for agent_name in self.agents}
        return obs, infos

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

    def _get_obs(self) -> ObsDict:
        """
        Determine the observations for all agents.

        :return: A dictionary of observations for each agent, indexed by the
            agent's name. Each observation is a dataclass containing all
            (global *and* local) metrics. Global and local observations
            can also be obtained through the
            :py:meth:`~smartgrid.observation.base_observation.Observation.get_global_observation`
            and :py:meth:`~smartgrid.observation.base_observation.Observation.get_local_observation`
            methods.
        """
        return {
            agent.name: self.observation_manager.compute(self.world, agent)
            for agent in self.world.agents
        }

    def _get_reward(self) -> RewardsDict:
        """
        Determine the reward for each agent.

        Rewards describe to which degree the agent's action was appropriate,
        w.r.t. moral values. These moral values are encoded in the reward
        function(s), see :py:mod:`smartgrid.rewards` for more details on them.

        Reward functions may comprise multiple objectives. In such cases, they
        can be aggregated so that the result is a single float (which is used
        by most of the decision algorithms).
        This behaviour (whether to aggregate, and how to aggregate) is
        controlled by an optional wrapper, see
        :py:class:`~smartgrid.wrappers.reward_aggregator.RewardAggregator`
         for details.

        :return: A dictionary of rewards, one element per agent. The element
            itself is a dict which contains at least one reward, indexed by the
            reward's name.
        """
        return {
            agent.name: self.reward_calculator.compute(self.world, agent)
            for agent in self.world.agents
        }

    def _get_info(self, rewards: RewardsDict) -> InfoDict:
        """
        Return additional information on the world (for the current time step).

        Information (currently) contain only the rewards, for each agent.

        :param rewards: The dictionary of rewards, one for each agent.
            As multiple reward functions can be used, rewards are represented
            as dictionaries themselves, indexed by the reward function's name.

        :return: A dictionary of additional information, indexed by the agent's
            name. Each element is itself a dictionary that currently contains
            only the agent's reward, indexed by ``'reward'``.
        """
        return {
            agent_name: {
                'reward': rewards[agent_name]
            }
            for agent_name in self.agents
        }

    @property
    def observation_shape(self):
        """The shape, i.e., number of dimensions, of the observation space."""
        return self.observation_manager.shape

    @property
    def agents(self) -> List[AgentID]:
        """
        The list of agents' *names* contained in the environment (world).

        .. warning:: As per the PettingZoo API, and contrary to what the name
            suggests, this returns the agents' *names* (IDs), not the agents
            themselves. Please see :py:meth:`~.get_agent` to get an Agent
            from its name.
        """
        # PettingZoo requires this to be a list rather than an Iterable
        return list(self.world.agents_by_name.keys())

    def get_agent(self, agent_name: AgentID) -> Agent:
        """
        Return an agent from its name (ID).

        :param agent_name: The name of the requested agent.
        """
        return self.world.agents_by_name[agent_name]

    @property
    def num_agents(self) -> int:
        """The number of agents currently living in the environment."""
        return len(self.world.agents_by_name)

    def observation_space(self, agent_name: AgentID) -> Space:
        """
        Return the observation space of a specific agent, identified by its name.

        :param agent_name: The name of the desired :py:class:`~smartgrid.agents.agent.Agent`.
            In practice, it does not impact the result, as all Agents use the
            same observation space.

        :return: An instance of :py:class:`gymnasium.spaces.Box` indicating
            the number of dimensions of an observation, as well as the ``low``
            and ``high`` bounds for each dimension.
        """
        return self.observation_spaces[agent_name]

    def action_space(self, agent_name: AgentID) -> Space:
        """
        Return the action space of a specific agent, identified by its name.

        :param agent_name: The name of the desired :py:class:`~smartgrid.agents.agent.Agent`.
            It must correspond to an existing Agent in the current World, i.e.,
            an agent in the :py:attr:`~smartgrid.world.World.agents` list.

        :return: An instance of :py:class:`gymnasium.spaces.Box` indicating
            the number of dimensions of actions (parameters), as well as the
            ``low`` and ``high`` bounds for each dimension.
        """
        return self.action_spaces[agent_name]

from typing import List

from gymnasium import Env

from smartgrid.environment import SmartGrid
from smartgrid.world import World
from smartgrid.agents import DataOpenEIConversion, Agent, comfort
from smartgrid.util import RandomEnergyGenerator
from smartgrid.observation import ObservationManager
from smartgrid.rewards import Reward
from smartgrid.rewards.numeric.differentiated import AdaptabilityThree
from smartgrid.wrappers import SingleRewardAggregator


def make_basic_smartgrid(
    rewards: List[Reward] = None,
    max_step: int = 10_000,
) -> Env:
    """
    Defines a "basic" scenario, a helper method to instantiate a SmartGrid env.

    This method limits the available parameters, and hence, the possible
    customization. It is used to simplify the creation of an environment.
    This basic environment is configured with:

    * 20 agents with the *Household* profile, 5 agents with the *Office*
      profile, and 1 agent with the *School* profile.
    * A :py:class:`smartgrid.util.RandomEnergyGenerator` which provides
      each time step between 80% and 120% of the agents' total energy need.
    * The :py:class:`smartgrid.rewards.numeric.differentiated.AdaptabilityThree`
      reward function, whose definition changes at t=2000 and t=6000 (to
      force agents to adapt).
    * The default :py:class:`smartgrid.observation.ObservationManager` to
      determine observations from the current state of the environment.

    Users that desire full control over the environment creation, e.g., to
    experiment with various scenarii, should instead manually create the
    environment "from scratch", as explained in the documentation. They
    may take inspiration from this method's content to do so.

    :param rewards: The list of reward functions to use (see the
        :py:mod:`smartgrid.rewards` package for a list of available reward
        functions. Traditionally, most users will want to use a single
        reward function (*single-objective* reinforcement learning), but
        this environment supports *multi-objective* reinforcement learning
        if desired. By default, the :py:class:`.AdaptabilityThree` reward
        function is used.

    :param max_step: The maximum number of steps to simulate in the environment.
        By default, a maximum number of ``10_000`` steps are allowed; however,
        the environment can still be used after this amount, but it will raise
        a warning. This is mainly used to control the *interaction loop*
        automatically through the *terminated* and *truncated* values.

    :return: An instance of a :py:class:`.SmartGrid` env.
        This instance must be, as per the Gymnasium framework, ``reset``
        before it can be used. The instance is wrapped in a
        :py:class:`.RewardAggregator` in order to receive single-objective
        rewards. To directly access the underlying env, use the
        :py:attr:`unwrapped` property.
    """

    # 1. Load the data (Agents' Profiles)
    converter = DataOpenEIConversion()
    # FIXME: what happens when the package is imported?
    #  This relative path will certainly not work.
    converter.load('Household',
                   './data/openei/profile_residential_annually.npz',
                   comfort.flexible_comfort_profile)
    converter.load('Office',
                   './data/openei/profile_residential_annually.npz',
                   comfort.neutral_comfort_profile)
    converter.load('School',
                   './data/openei/profile_residential_annually.npz',
                   comfort.strict_comfort_profile)

    # 2. Create Agents
    agents = []
    for i in range(20):
        agents.append(
            Agent(f'Household{i+1}', converter.profiles['Household'])
        )
    for i in range(5):
        agents.append(
            Agent(f'Office{i+1}', converter.profiles['Office'])
        )
    agents.append(
        Agent(f'School1', converter.profiles['School'])
    )

    # 3. Create the World
    generator = RandomEnergyGenerator()
    world = World(agents, generator)

    # 4. Choose the reward function(s) to use
    if rewards is None:
        rewards = [AdaptabilityThree()]

    # 5. Create the Env (Smart Grid simulator)
    simulator = SmartGrid(
        world,
        rewards,
        max_step,
        ObservationManager()
    )

    # 6. Wrap the Env to receive a single (scalar) reward instead of a dict.
    simulator = SingleRewardAggregator(simulator)

    return simulator

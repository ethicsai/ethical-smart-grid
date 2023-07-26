"""
A few pre-defined scenarii to quickly launch simulations without coding anything.

These scenarii are, contrary to the rest of the code, not meant to be easily
extensible or flexible, just to provide a few pre-defined environment setups
that can be leveraged in a single import, rather than having to define them
"by hand". In this sense, they are by no means exhaustive.

They offer the additional advantage of serving as examples for writing your
own custom setups: please feel free to read their source code and copy-paste
at will!

It currently contains the following scenarii:

- :py:func:`~.small_daily_scenario` : Small neighborhood, daily profiles of
  consumption, using the OpenEI dataset.
- :py:func:`~.medium_daily_scenario` : Medium neighborhood, daily profiles of
  consumption, using the OpenEI dataset.
- :py:func:`~.medium_annual_scenario` : Medium neighborhood, annual profiles of
  consumption, using the OpenEI dataset.

(The "small + annual" is already handled in the "basic" (default) environment,
see :py:func:`~smartgrid.make_env.make_basic_smartgrid`.)

To use one of these scenarii, simply instantiate a new environment by
invoking one of the functions, and use this environment through the classic
interaction loop. For example:

.. code-block:: Python

    from smartgrid.scenarii import medium_annual_scenario
    env = medium_annual_scenario(max_step=10)

    done = False
    obs = env.reset()
    while not done:
        actions = [
            agent.profile.action_space.sample()
            for agent in env.unwrapped.agents
        ]
        obs, rewards, terminated, truncated, _ = env.step(actions)
        done = all(terminated) or all(truncated)

"""

from typing import Optional

from smartgrid import World, SmartGrid
from smartgrid.agents import DataOpenEIConversion, Agent
from smartgrid.agents.profile import comfort
from smartgrid.make_env import find_profile_data
from smartgrid.observation import ObservationManager
from smartgrid.rewards.numeric.differentiated import AdaptabilityThree
from smartgrid.util import RandomEnergyGenerator
from smartgrid.wrappers import SingleRewardAggregator


def small_daily_scenario(max_step: Optional[int] = 10_000):
    """
    A scenario with a *small* neighborhood and *daily* profiles of consumption.

    *Small* refers to: 20 Households, 5 Offices, and 1 School.
    *Daily* means that the consumption profiles are aggregated over a single
    day, and provide data for each hour of this "average day". It contains
    fewer changes than the "full" dataset (annual), e.g., no seasonal change.

    It uses the *OpenEI* dataset of profiles, and the *Adaptability3* reward
    function.

    :param max_step: The maximal number of time steps that the simulation
        should run for. By default, we assume 10,000 time steps, which can be
        quite long but provide enough data. It can also be set to `None` to
        let the *interaction loop* control when to stop (warning: the
        environment will never stop by itself in this case! Do not use the
        `done` value to avoid running an infinite loop!).

    :return: An environment pre-configured and ready to use (except for the
        `reset` call!).
    :rtype: gymnasium.Env
    """

    # 1. Load the data (Agents' Profiles)
    converter = DataOpenEIConversion()
    converter.load('Household',
                   find_profile_data('openei', 'profile_residential_daily.npz'),
                   comfort.flexible_comfort_profile)
    converter.load('Office',
                   find_profile_data('openei', 'profile_office_daily.npz'),
                   comfort.neutral_comfort_profile)
    converter.load('School',
                   find_profile_data('openei', 'profile_school_daily.npz'),
                   comfort.strict_comfort_profile)

    # 2. Create Agents
    agents = []
    for i in range(20):
        agents.append(
            Agent(f'Household{i + 1}', converter.profiles['Household'])
        )
    for i in range(5):
        agents.append(
            Agent(f'Office{i + 1}', converter.profiles['Office'])
        )
    agents.append(
        Agent(f'School1', converter.profiles['School'])
    )

    # 3. Create the World
    generator = RandomEnergyGenerator()
    world = World(agents, generator)

    # 4. Choose the reward function(s) to use
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


def medium_daily_scenario(max_step: Optional[int] = 10_000):
    """
    A scenario with a *medium* neighborhood and *daily* profiles of consumption.

    *Medium* refers to: 100 Households, 19 Offices, and 1 School.
    *Daily* means that the consumption profiles are aggregated over a single
    day, and provide data for each hour of this "average day". It contains
    fewer changes than the "full" dataset (annual), e.g., no seasonal change.

    It uses the *OpenEI* dataset of profiles, and the *Adaptability3* reward
    function.

    :param max_step: The maximal number of time steps that the simulation
        should run for. By default, we assume 10,000 time steps, which can be
        quite long but provide enough data. It can also be set to `None` to
        let the *interaction loop* control when to stop (warning: the
        environment will never stop by itself in this case! Do not use the
        `done` value to avoid running an infinite loop!).

    :return: An environment pre-configured and ready to use (except for the
        `reset` call!).
    :rtype: gymnasium.Env
    """

    # 1. Load the data (Agents' Profiles)
    converter = DataOpenEIConversion()
    converter.load('Household',
                   find_profile_data('openei', 'profile_residential_daily.npz'),
                   comfort.flexible_comfort_profile)
    converter.load('Office',
                   find_profile_data('openei', 'profile_office_daily.npz'),
                   comfort.neutral_comfort_profile)
    converter.load('School',
                   find_profile_data('openei', 'profile_school_daily.npz'),
                   comfort.strict_comfort_profile)

    # 2. Create Agents
    agents = []
    for i in range(80):
        agents.append(
            Agent(f'Household{i + 1}', converter.profiles['Household'])
        )
    for i in range(19):
        agents.append(
            Agent(f'Office{i + 1}', converter.profiles['Office'])
        )
    agents.append(
        Agent(f'School1', converter.profiles['School'])
    )

    # 3. Create the World
    generator = RandomEnergyGenerator()
    world = World(agents, generator)

    # 4. Choose the reward function(s) to use
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


def medium_annual_scenario(max_step: Optional[int] = 10_000):
    """
    A scenario with a *medium* neighborhood and *annual* profiles of consumption.

    *Medium* refers to: 100 Households, 19 Offices, and 1 School.
    *Annual* means that the consumption profiles are not aggregated, and provide
    data for each hour of each day of a full year. It thus contains more changes
    than the aggregated dataset (daily), e.g., with seasonal changes between
    summer and winter.

    It uses the *OpenEI* dataset of profiles, and the *Adaptability3* reward
    function.

    :param max_step: The maximal number of time steps that the simulation
        should run for. By default, we assume 10,000 time steps, which can be
        quite long but provide enough data. It can also be set to `None` to
        let the *interaction loop* control when to stop (warning: the
        environment will never stop by itself in this case! Do not use the
        `done` value to avoid running an infinite loop!).

    :return: An environment pre-configured and ready to use (except for the
        `reset` call!).
    :rtype: gymnasium.Env
    """

    # 1. Load the data (Agents' Profiles)
    converter = DataOpenEIConversion()
    converter.load('Household',
                   find_profile_data('openei', 'profile_residential_annually.npz'),
                   comfort.flexible_comfort_profile)
    converter.load('Office',
                   find_profile_data('openei', 'profile_office_annually.npz'),
                   comfort.neutral_comfort_profile)
    converter.load('School',
                   find_profile_data('openei', 'profile_school_annually.npz'),
                   comfort.strict_comfort_profile)

    # 2. Create Agents
    agents = []
    for i in range(80):
        agents.append(
            Agent(f'Household{i + 1}', converter.profiles['Household'])
        )
    for i in range(19):
        agents.append(
            Agent(f'Office{i + 1}', converter.profiles['Office'])
        )
    agents.append(
        Agent(f'School1', converter.profiles['School'])
    )

    # 3. Create the World
    generator = RandomEnergyGenerator()
    world = World(agents, generator)

    # 4. Choose the reward function(s) to use
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

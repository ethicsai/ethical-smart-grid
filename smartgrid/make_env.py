"""
Functions that can be used to create a "basic" (or standard) environment.
"""


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

import os


def find_profile_data(dataset: str, filename: str) -> str:
    """
    Finds a data file from the data folder.

    The ``data`` folder can be either:

    - accessible through a relative path, e.g., because the source code was
      cloned (``./data/path/to/the/desired/file``).
    - accessible through imports, e.g., because the package was installed.

    How to access files in this folder depends on the actual case; this function
    (hopefully) solves this problem, by checking first whether the files are
    directly accessible (through relative paths). If they are not,
    :py:mod:`importlib.resources` is used to get a path to the file through the
    package.

    This function may raise a ``DeprecationWarning`` when accessing
    files through the package on Python3.11+. It is due to relying on the
    :py:func:`importlib.resources.path` function, which is deprecated since
    Python 3.11. As we support up to Python 3.7, and as the function is
    only available since Python 3.9, it is still used instead of the more
    recent :py:func:`importlib.resources.files`.

    :param dataset: The name of the folder containing the desired file, within
        the ``data`` folder. For example, ``openei`` to access the OpenEI
        dataset. This dataset cannot be nested, e.g., ``dataset/subdataset``
        or ``dataset.subdataset`` will not work.

    :param filename: The name of the desired file, within the ``dataset``
        folder.

    :return: A path to the desired file.
    """
    # Equivalent to `./data/dataset/filename`, but OS-agnostic.
    relative_path = os.path.join('data', dataset, filename)
    if os.path.isfile(relative_path):
        # Easy mode: the file is directly accessible through a relative path!
        return relative_path
    else:
        # Hard mode: need to access it through the package (importlib).
        # Also, importlib returns a context, so we need to get the path
        # and exit the context when Python terminates.
        import importlib_resources
        from contextlib import ExitStack
        import atexit
        # Handle the context
        file_manager = ExitStack()
        # Close the context when Python terminates
        atexit.register(file_manager.close)
        # `path` is deprecated since Python3.9, but we support up to 3.7,
        # so we still use `path`. There is not much benefit to check whether
        # the new `files` API is available and use it instead.
        ctx = importlib_resources.path(f'smartgrid.data.{dataset}', filename)
        path = file_manager.enter_context(ctx)
        # Convert the PosixPath to a simple str for easier usage.
        return str(path)


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

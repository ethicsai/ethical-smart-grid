Custom scenario
===============

This Smart Grid simulator was designed to support various experiments,
including in terms of agents (number, profiles), physical constraints in the
world (available energy), and reward functions.
We call the combination of these elements a *scenario*, and we describe here
how to fully customize a scenario.

For each of these elements, we give a succinct description; for a better,
more complete understanding of how they work, please refer to their API
documentation.

Agents' profiles
----------------

Agents have some common characteristics, such as their personal battery
capacity, how much energy they need each step, how much energy they produce,
how they determine their comfort based on their need and consumption.
To simplify the creation of agents and reduce resources (memory and
computations), these characteristics are grouped and shared in *Profiles*.

An :py:class:`~smartgrid.agents.profile.agent_profile.AgentProfile`
can be loaded from data files (see, e.g., the ``data/openei`` folder);
to do so, it is necessary to use a
:py:class:`~smartgrid.agents.data_conversion.DataConversion` object.

For example:

.. code-block:: Python

    from smartgrid.agents import DataOpenEIConversion
    from smartgrid.agents import comfort

    # Create a converter specialized for the `data/openei` files.
    converter = DataOpenEIConversion()

    # Load agents' profiles, using the data files.
    converter.load(
        name='Household',  # Profile name -- a unique ID
        data_path='./data/openei/profile_residential_annually.npz',  # Data file
        comfort_fn=comfort.flexible_comfort_profile  # Comfort function
    )
    converter.load(
        'Office',
        './data/openei/profile_office_annually.npz',
        comfort.neutral_comfort_profile
    )
    converter.load(
        'School',
        './data/openei/profile_school_annually.npz',
        comfort.strict_comfort_profile
    )

    # Profiles can be accessed through the `profiles` attribute, and are indexed
    # by their ID.
    profile_household = converter.profiles['Household']
    profile_office = converter.profiles['Office']
    profile_school = converter.profiles['School']

You can use the converter object to load any profile you desire, and use these
profiles to instantiate :py:class:`~smartgrid.agents.agent.Agent`\ s.

Energy generator
----------------

The :py:class:`~smartgrid.util.available_energy.EnergyGenerator`
is used to determine, at each time step, the amount of available energy in the
world.
Several implementations are available, e.g., the
:py:class:`~smartgrid.util.available_energy.RandomEnergyGenerator`,
the :py:class:`~smartgrid.util.available_energy.ScarceEnergyGenerator`,
or the :py:class:`~smartgrid.util.available_energy.GenerousEnergyGenerator`.
They "generate" a random amount of energy based on the total need of all agents
at the current time step.
Another implementation is the
:py:class:`~smartgrid.util.available_energy.RealisticEnergyGenerator`,
which uses a dataset of productions per time step to determine the amount.

For example, using a random generator:

.. code-block:: Python

    from smartgrid.util import RandomEnergyGenerator

    # This generator will generate between 75% and 110% of the agents' total need
    # at each step.
    generator = RandomEnergyGenerator(
        lower_proportion=0.75,
        upper_proportion=1.1
    )

    # Example with current_need = 10_000 Wh.
    amount = generator.generate_available_energy(
        current_need=10_000,
        # The other values are not important for this generator.
        current_step=0,
        min_need=0,
        max_need=100_000
    )
    assert 0.75 * 10_000 <= amount < 1.10 * 10_000

Another example, using the realistic generator:

.. code-block:: Python

    from smartgrid.util import RealisticEnergyGenerator

    # The dataset (source of truth) for energy production at each time step.
    # This dataset means that, at t=0, 80% of the agents' maximum need will be
    # available; at t=1, 66% of their maximum need; and at t=2, 45%.
    # Subsequent time steps will simply cycle over this array, e.g., t=3 is
    # the same as t=0.
    data = [0.80, 0.66, 0.45]
    generator = RealisticEnergyGenerator(data=data)

    # Example with current_need = 10_000 Wh.
    amount = generator.generate_available_energy(
        max_need=100_000,
        current_step=0,
        # The other values are not important for this generator.
        current_need=10_000,
        min_need=0
    )
    assert amount == int(100_000 * data[0])

World
-----

The :py:class:`~smartgrid.world.World` represents a simulated "physical" world.
It handles the physical aspects: agents, available energy, and updates through
agents' actions.

The world is instantiated from a list of agents, and an energy generator:

.. code-block:: Python

    from smartgrid import World
    from smartgrid.agents import Agent

    # We assume that the variables instantiated above are available,
    # especially the `converter` (with loaded profiles) and the `generator`.

    # Create the agents, based on loaded profiles.
    agents = []
    for i in range(5):
        agents.append(
            Agent(
                name=f'Household{i+1}',  # Unique name -- recommended to use profile + index
                profile=converter.profiles['Household']  # Agent Profile
            )
        )
    for i in range(3):
        agents.append(
            Agent(f'Office{i+1}', profile_office)
        )

    # Create the world, with agents and energy generator.
    world = World(
        agents=agents,
        energy_generator=generator
    )

At this point, we have a usable world, able to simulate a smart grid, and to
update itself when agents take actions.
(It is even usable as-is, if you are not interested in Reinforcement Learning!)
However, to benefit from the RL *interaction loop* (observations, actions, rewards),
we have to create an Environment.

Reward functions
----------------

Reward functions dictate what is the agents' expected behaviour.
Several have been implemented and are directly available; they target different
ethical considerations, such as equity, maximizing comfort, etc.
Please refer to the :py:mod:`rewards <smartgrid.rewards>` module for a detailed
list.

A particularly interesting reward function is
:py:class:`~smartgrid.rewards.numeric.differentiated.adaptability.AdaptabilityThree`:
its definition evolves as the time steps increase, which forces agents to adapt
to changing ethical considerations and objectives.

To use it, simply import it and create an instance:

.. code-block:: Python

    from smartgrid.rewards.numeric.differentiated import AdaptabilityThree

    rewards = [AdaptabilityThree()]

.. note::
    The environment has (partial) support for *Multi-Objective* RL (MORL),
    hence the use of a list of rewards.
    When using "traditional" (*single-objective*) RL algorithms, make sure to
    specify only 1 reward function, and to use a wrapper that aggregates several
    rewards into a single scalar number.

SmartGrid Env
-------------

Finally, the :py:class:`~smartgrid.environment.SmartGrid` class
represents the link with Gymnasium's standard, by extending the
:py:class:`~gymnasium.core.Env` class.
It is responsible for providing observations at each time step, receiving
actions, and computing the rewards based on observations and actions.

.. code-block:: Python

    from smartgrid import SmartGrid

    env = SmartGrid(
        world=world,
        rewards=rewards
    )

Maximum number of steps
^^^^^^^^^^^^^^^^^^^^^^^

By default, the environment does not terminate: it is not episodic. The
simulation will run as long as the *interaction loop* continues. It is possible
to set a maximum number of steps, so that the environment will signal, through
its ``truncated`` return value, that it should stop. This can be especially
useful when using specialized learning libraries that are built to automatically
check the ``terminated`` and ``truncated`` return values.

To do so, simply set the parameter when creating the instance:

.. code-block:: Python

    env = SmartGrid(
        world=world,
        rewards=rewards,
        max_step=10_000
    )

After ``max_step`` steps have been done, the environment can still be used,
but it will emit a warning.

Single- or multi-objective
^^^^^^^^^^^^^^^^^^^^^^^^^^

If only 1 reward function is used, and *single-objective* learning algorithms
are targeted, the env may be wrapped in a specific class that returns a single
(scalar) reward instead of a dict:

.. code-block:: Python

    from smartgrid.wrappers import SingleRewardAggregator

    env = SingleRewardAggregator(env)

This simplifies the usage of the environment for most cases. When dealing with
multiple reward functions, other aggregators such as the
:py:class:`~smartgrid.wrappers.reward_aggregator.WeightedSumRewardAggregator`,
or the :py:class:`~smartgrid.wrappers.reward_aggregator.MinRewardAggregator`
can be used instead. To use *multi-objective* learning algorithms, which
receive several rewards each step, simply avoid wrapping the base environment.

When the environment is wrapped, the base environment can be obtained through
the :py:obj:`~gymnasium.Wrapper.unwrapped` property. Gymnasium
wrappers should allow access to any (public) attribute automatically:

.. code-block:: Python

   smartgrid = env.unwrapped
   n_agent = env.n_agent  # Note that `n_agent` is not defined in the wrapper!
   assert n_agent == smartgrid.n_agent

The interaction loop
^^^^^^^^^^^^^^^^^^^^

The Env is now ready for the *interaction loop*!

If a maximum number of step has been specified, the traditional ``done`` loop
can be used:

.. code-block:: Python

    done = False
    obs_n = env.reset()
    while not done:
        # Implement your decision algorithm here
        actions = [
            agent.profile.action_space.sample()
            for agent in env.agents
        ]
        obs_n, rewards_n, terminated_n, truncated_n, info_n = env.step(actions)
        done = all(terminated_n) or all(truncated_n)
    env.close()

Otherwise, the env termination must be handled by the interaction loop itself:

.. code-block:: Python

    max_step = 50
    obs_n = env.reset()
    for _ in range(max_step):
        # Implement your decision algorithm here
        actions = [
            agent.profile.action_space.sample()
            for agent in env.agents
        ]
        # Note that we do not need the `terminated` nor `truncated` values here.
        obs_n, rewards_n, _, _, info_n = env.step(actions)
    env.close()

Both ways are completely equivalent: use one or the other at your convenience.

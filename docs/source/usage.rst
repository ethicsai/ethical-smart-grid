Usage
=====

|project_name| provides a `Gymnasium <https://gymnasium.farama.org/>`_
environment for multi-agent reinforcement learning of *ethically-aligned*
behaviours, i.e., behaviours that take into account ethical considerations.

The environment is available as a Python package, and follows the Gymnasium API
as close as possible, such that it can be used with any Gymnasium-compliant
learning algorithm with little to no modification.

Installation
------------

To use this simulator, you may either:

* clone the repository: ``git clone https://github.com/ethicsai/ethical-smart-grid``
* or install the package from PyPi: ``pip install ethical-smart-grid``

Cloning is the recommended way to get the up-to-date version, and is easier if
you intend to implement new algorithms and/or extend the simulator.
Cloning also allows you to refer to data files by relative paths, whereas
downloading the package from PyPi requires to refer to data files as package
resources.

Running a simulation
--------------------

The environment is designed to allow various *scenarii* and to be highly
configurable and extensible.
To simplify the creation of a basic environment, the method
:py:meth:`smartgrid.make_basic_smartgrid() <smartgrid.make_env.make_basic_smartgrid>`
is made available.
This method is also used to register the Smart Grid environment with Gymnasium,
using the :py:func:`gym.make() <gymnasium.make>` method.

To create an environment, type in a Python console:

.. code-block:: Python

    from smartgrid import make_basic_smartgrid
    env = make_basic_smartgrid()

Or, equivalently:

.. code-block:: Python

    import gymnasium as gym
    import smartgrid

    env = gym.make('EthicalSmartGrid-v0')

Then, the environment can be used through the standard *interaction loop*:

.. code-block:: Python

    done = False
    obs = env.reset()
    while not done:
        # Replace the `actions` array with your own learning algorithm here
        actions = [
            agent.profile.action_space.sample()
            for agent in env.agents
        ]
        obs, reward_n, terminated_n, truncated_n, info_n = env.step(actions)
        done = all(terminated_n) or all(truncated_n)


In order to fully customize the environment setup, e.g., to control the
number and profiles of agents, the available energy in the world, the reward
function, etc., please refer to the :doc:`custom_scenario` documentation page.

The environment can also be extended to add your own components or replace
existing ones; please refer to :doc:`/extending/index` to do so.

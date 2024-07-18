Adding a new model
==================

One of the principal goals of this simulator is to be able to compare various
learning algorithms (similarly to PettingZoo's environments).
This page describes how to implement another learning algorithm (i.e., *model*).

Models interact with the :py:class:`SmartGrid <smartgrid.environment.SmartGrid>`
through the *interaction loop*:

.. code-block:: Python

    from smartgrid import make_basic_smartgrid

    env = make_basic_smartgrid()
    obs, _ = env.reset()
    max_step = 10  # Can also be 10_000, ...
    for step in range(max_step)
        actions = model.forward(obs)  # Add your model here!
        obs, rewards, _, _, infos = env.step(actions)
        model.backward(obs, rewards)  # And here!

.. note::
    We have used ``forward`` and ``backward`` as method names for the mode,
    because they are commonly-found when dealing with (Deep) Neural Networks.
    However, models do not need to follow this exact naming convention: they
    exist outside of the environment. Anything can be used, even pure functions
    instead of classes, as long as the environment receives the ``actions``,
    and the model takes the ``obs`` and ``rewards`` as input.

Forward: producing actions
--------------------------

We first describe the *forward* step, which consists of producing actions
based on observations.
Let us consider a model that returns random actions; to simplify, the model
takes into account all agents at the same time, but different models can be
used for different agents.

.. code-block:: Python

    import numpy as np

    from smartgrid.util import interpolate

    class CustomModel:

        def __init__(self, env):
            self.env = env

        def forward(obs):
            # `obs` is a dict mapping each agent name to its observations.
            # Agent observations are namedtuples that can be printed for
            # easier human readability and debugging, or transformed to
            # numpy arrays (with `np.asarray`) for easier handling by Neural
            # Networks.

            # The env expects a dict mapping each agent name to its desired action.
            # Here, we simply create a random action for each agent, with Numpy.
            agent_actions = {}
            for agent_name in self.env.agents:
                # `obs[agent_name]` are the agent's observations
                # We need the action's number of dimensions. It should be 6,
                # but the SmartGrid can be extended and so it's better to avoid
                # hard-coding it.
                agent_action_space = self.env.action_space(agent_name)
                agent_action_nb_dimensions = agent_action_space.shape[0]
                action = np.random.random(agent_action_nb_dimensions)
                # `action` is a ndarray of 6 values in [0,1].
                # Most learning algorithms will handle values in [0, 1], but the
                # SmartGrid env may expect actions in a different space, depending
                # on the agent's profile. We can use `interpolate` to transform.
                action = interpolate(
                    value=action,
                    old_bounds=[(0,1)] * agent_action_nb_dimensions,
                    new_bounds=list(zip(agent_action_space.low, agent_action_space.high))
                )
                agent_actions[agent_name] = action

            # At this point, `agent_actions` is a dict of actions (ndarrays), one
            # element for each agent.
            return agent_actions

As we need to know the *action space* of the various agents, we pass the ``env``
instance to the constructor of the model.

Backward: learning from rewards
-------------------------------

Once actions have been executed in the environment, new observations and rewards
are produced, and sent to the model in the *backward* step, so it can improve
its decision-making, e.g., by updating weights in a neural network.

In this example, as we use completely random values, we do not need the rewards,
but we will illustrate the ``backward`` method anyway:

.. code-block:: Python

    class CustomModel:

    # (...) code from previous section

    def backward(self, new_obs, rewards):
        for agent_name in self.env.agents:
            # `new_obs` is a dict of observations, one element for each agent.
            agent_obs = new_obs[agent_name]
            # `rewards` is also a dict; each element can be:
            # - a scalar (single value) if the SmartGrid env has a single reward
            #   function (single-objective);
            # - a dict mapping reward names to their values, if the env has
            #   multiple reward functions (multi-objective).
            agent_reward = rewards[agent_name]

.. warning::
    If you do not use a :py:class:`~smartgrid.wrappers.reward_aggregator.RewardAggregator`
    wrapper over the environment, the ``rewards`` object will be a list of dicts,
    containing the different rewards for each agents (multiple rewards),
    instead of a list of single rewards! In this case, the dict is indexed by
    the reward functions' names. By default, this wrapper is used.

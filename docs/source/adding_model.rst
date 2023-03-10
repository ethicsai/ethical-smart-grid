Adding a new model
==================

One of the principal goals of this simulator is to be able to compare various
learning algorithms (similarly to Gymnasium's environments).
This page describes how to implement another learning algorithm (i.e., *model*).

Models interact with the :py:class:`SmartGrid <smartgrid.environment.SmartGrid>`
through the *interaction loop*:

.. code-block:: Python

    from smartgrid import make_basic_smartgrid

    env = make_basic_smartgrid()
    obs = env.reset()
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
            # `obs` is a dict containing:
            # - `global`: an instance of GlobalObservation;
            # - `local`: a list of instances of LocalObservations, one per agent.
            # To reconstruct the observations per agent, a for loop can be used:
            obs_per_agent = [
                np.concatenate((
                    obs['local'][i],
                    obs['global'],
                ))
                for i in range(self.env.n_agent)
            ]
            # Then, each element of `obs_per_agent` can be used for the specific agent.
            # Here, we simply use random.
            agent_actions = []
            for i in range(self.env.n_agent):
                # We need the number of dimensions of the action. It should be 6, but
                # it's better to avoid hard-coding it.
                agent_action_space = self.env.action_space[i]
                agent_action_nb_dimensions = agent_action_space.shape[0]
                action = np.random.random(agent_action_nb_dimensions)
                # `action` is a ndarray of 6 values in [0,1].
                # Most learning algorithms will handle values in [0,1], but the
                # SmartGrid env actually expects actions in a different space,
                # depending on the agent's profile. We can use `interpolate`
                # to make the transformation.
                action = interpolate(
                    value=action,
                    old_bounds=[(0,1)] * agent_action_nb_dimensions,
                    new_bounds=list(zip(agent_action_space.low, agent_action_space.high))
                )
                agent_actions.append(action)
            # At this point, `agent_actions` is a list of actions (ndarrays), one
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
        # `new_obs` has the same shape as `obs` in `forward`: `global` and `local`.
        new_obs_per_agent = [
            np.concatenate((
                new_obs['local'][i],
                new_obs['global'],
            ))
            for i in range(self.env.n_agent)
        ]
        # `rewards` will be usually a list of scalar values, one per agent

.. warning::
    If you do not use a :py:class:`~smartgrid.wrappers.reward_aggregator.RewardAggregator`
    wrapper over the environment, the ``rewards`` object will be a list of dicts,
    containing the different rewards for each agents (multiple rewards),
    instead of a list of single rewards! In this case, the dict is indexed by
    the reward functions' names. By default, this wrapper is used.

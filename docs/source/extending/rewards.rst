Reward functions
================

Reward functions encode the moral values that agents' behaviours should be
aligned with. Their goal is to reward (positively or negatively) agents, by
judging to which degree their actions are indeed aligned with the moral values.

New reward functions can be implemented in order to give new incentives to
learning agents, and to encourage different behaviours.
Traditionally, reward functions are defined as a purely mathematical function,
but any kind of computation that returns a floating-point number can be used.

Implementing a new reward function is as simple as extending the
:py:class:`~smartgrid.rewards.reward.Reward` class, and overriding its
:py:meth:`~smartgrid.rewards.reward.Reward.calculate` method.
This method takes as parameters the :py:class:`~smartgrid.world.World`
and the :py:class:`~smartgrid.agents.agent.Agent` that is currently
judged, and must return a float.

Most of the time, reward functions will output numbers in the ``[0,1]`` range;
yet, it is not strictly required. However, if multiple rewards are used at the
same time (*multi-objective reinforcement learning*), users will want to make
sure that they have similar ranges, otherwise the agents could be biased towards
one or another reward function.

For example, we will implement a simple function that encourages agents to
fill their personal battery.

.. code-block:: Python

    from smartgrid.rewards import Reward

    class FillBattery(Reward):

        def calculate(self, world, agent):
            # `storage_ratio` is the current ratio of energy in the battery
            # compared to the battery's maximal capacity, where 0 indicates
            # the battery is empty, whereas 1 indicates the battery is full.
            # This fictitious reward function should encourage agents to fill
            # their batteries, thus give high reward to full batteries.
            # Returning the ratio itself is a very simple way to do so!
            return agent.storage_ratio

Such a simple reward function could be implemented by a simple Python function,
however, using a class for reward functions allows more complex mechanisms,
e.g., to memorize previous elements.
Let us consider a second example, in which we want to encourage the agent
to gain money by rewarding the difference with the previous step.

.. code-block:: Python

    from smartgrid.rewards import Reward

    class GainMoney(Reward):

        def __init__(self):
            super().__init__()
            self.previous_payoffs = {}

        def calculate(self, world, agents):
            # Get (or use default) the payoff at the last step.
            previous_payoff = self.previous_payoffs.get(agent)
            if previous_payoff is None:
                previous_payoff = 0
            # The new payoff (in [0,1]).
            new_payoff = agent.payoff_ratio
            # Memorize the new payoff for the next step.
            self.previous_payoffs[agent] = new_payoff
            # When `new_payoff` > `previous_payoff`, the difference will be
            # positive, thus effectively rewarding agents when they have more
            # money than at the previous time step.
            reward = new_payoff - previous_payoff
            return reward

        def reset(self):
            self.previous_payoffs = {}

Note that, when such a *mutable state* is used within a reward function, the
reward function must override the :py:meth:`~smartgrid.rewards.reward.Reward.reset`
method, to reset the state. This ensures than, when the environment is reset,
reward functions can be used "as good as new".

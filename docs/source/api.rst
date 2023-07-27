API Reference
=============

.. image:: /images/architecture.drawio.png
   :target: /images/architecture.drawio.png
   :alt: Simplified class diagram representing the links between the important classes. Attributes and methods are not shown.

.. autosummary::
   :toctree: modules
   :recursive:

   smartgrid
   algorithms


To facilitate the exploration of the API, the pre-defined components of
interest are listed below, along with a short description.

- **Agents profiles and data conversion**

  - :py:mod:`~smartgrid.agents.profile.comfort` functions : describe how the
    agent's comfort should be computed, based on the consumption and need.

    - :py:func:`~smartgrid.agents.profile.comfort.flexible_comfort_profile` :
      A comfort function that is easy to satisfy: the curve quickly increases.

    - :py:func:`~smartgrid.agents.profile.comfort.neutral_comfort_profile` :
      A comfort function that increases "normally", the comfort follows more
      or less the ratio ``consumption / need``.

    - :py:func:`~smartgrid.agents.profile.comfort.strict_comfort_profile` :
      A comfort function that is difficult to satisfy: the consumption must
      be close to the need to see the comfort increase significantly.

  - :py:mod:`Data Conversion <smartgrid.agents.data_conversion>` : describe
    how to create an ``AgentProfile`` from raw data (dataset).

    - :py:class:`~smartgrid.agents.data_conversion.DataOpenEIConversion` :
      Creates profiles from the OpenEI dataset. Allows 3 kinds of buildings
      (*Households*, *Offices*, *Schools*) and 2 sources of needs (*daily* -
      aggregated and simplified, 24 data points ; *annual* - full dataset,
      24*365 data points). Agent production is simulated (no available data).

|

- **World setup**

  - :py:mod:`Energy Generators <smartgrid.util.available_energy>` : describe
    how much energy should be available each step, based on the agents' total
    need at this step, minimal and maximal need.

    - :py:class:`~smartgrid.util.available_energy.RandomEnergyGenerator` :
      The "basic" energy generator, simply returns a random value based on the
      agents' current total need. The bounds are configurable (e.g., between
      *80%* and *120%* of their current total need).

    - :py:class:`~smartgrid.util.available_energy.ScarceEnergyGenerator` :
      A random energy generator using *60%* and *80%* as bounds, i.e., there is
      never enough (100%) energy to satisfy all agents' needs. This forces
      conflict situations between agents.

    - :py:class:`~smartgrid.util.available_energy.GenerousEnergyGenerator` :
      A random energy generator using *100%* and *120%* as bounds, i.e., there
      is always enough energy to satisfy all agents' needs. This represents an
      easy scenario that can be combined with scarcity to see how agents
      perform after a change (e.g., how agents trained for scarcity perform
      during abundance, or conversely).

    - :py:class:`~smartgrid.util.available_energy.RealisticEnergyGenerator` :
      An energy generator that uses a dataset to determine how much energy
      is generated at each step. To ensure that it scales with the number of
      agents (and thus re-using datasets), this generator assumes that data
      correspond to percentages of agents' maximum needs. This generator can be
      used to introduce time-varying dynamics, e.g., less energy in winter than
      in summer because of less solar power.

|

- **Reward functions**

  - :py:mod:`Numerical reward functions <smartgrid.rewards.numeric>` : Reward
    functions that are based on a "mathematical formula", as is traditionally
    done in Reinforcement Learning.

    - :py:mod:`Difference Rewards <smartgrid.rewards.numeric.differentiated>` :
      Reward functions that are based on comparing the actual environment
      with an hypothetical environment in which the agent did not act. This
      gives us an idea of the agent's contribution.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.equity.Equity` :
        Focus on increasing the equity of inhabitants' comforts.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.over_consumption.OverConsumption` :
        Focus on reducing the over-consumption of agents, with respect to the
        quantity of available energy.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.multi_objective_sum.MultiObjectiveSum` :
        Focus on 2 objectives (increasing comfort, reducing over-consumption),
        and aggregates them with a simple sum.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.multi_objective_product.MultiObjectiveProduct` :
        Focus on 2 objectives (increasing comfort, reducing over-consumption),
        and aggregates by multiplying them.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.adaptability.AdaptabilityOne` :
        Focus on 3 objectives (increasing equity when t<3000, then average of
        increasing comfort and reducing over-consumption). The reward function
        thus "changes" as time goes on, and the new considerations are
        completely different.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.adaptability.AdaptabilityTwo` :
        Focus on 2 objectives (increasing equity when t<2000, then average of
        increasing equity and reducing over-consumption). The reward function
        thus "changes" as time goes on, but simply adds a new consideration
        to the existing one.

      - :py:class:`~smartgrid.rewards.numeric.differentiated.adaptability.AdaptabilityThree` :
        Focus on 3 objectives (increasing equity when t<2000, then average of
        increasing equity and reducing over-consumption when t<6000, then
        average of increasing equity, reducing over-consumption and increasing
        comfort). The reward function thus "changes" as time goes on, and
        adds several new considerations to the existing ones.

    - :py:mod:`Per-agent Rewards <smartgrid.rewards.numeric.per_agent>` :
      Reward functions that only consider the actual environment to compute
      the agent's contribution. They have similar objectives as the Difference
      Rewards, but are particularly useful when the learning algorithm itself
      has a mechanism to determine agents' contributions (e.g.,
      `COMA <https://arxiv.org/abs/1705.08926>`_).

      - :py:class:`~smartgrid.rewards.numeric.per_agent.comfort.Comfort` :
        Focus on increasing the agent's comfort. Can be used for self-interested
        agents, or in combination with other reward functions to create complex
        behaviours.

|

- **(Learning) algorithms**

  - :py:class:`~algorithms.qsom.qsom.QSOM` : the QSOM learning algorithm,
    based on the well-known Q-Learning and associated to 2 Self-Organizing
    Maps (SOMs) to handle multi-dimensional and continuous observations and
    actions.

  - :py:class:`~algorithms.naive.random_model.RandomModel` : a naive algorithm
    that performs purely random actions. It is provided to easily check that
    the environment is working, without having to fine-tune a learning
    algorithm, and coding the random decision directly. It can also be used as
    a baseline to compare other algorithms with, although it is a very low
    baseline.

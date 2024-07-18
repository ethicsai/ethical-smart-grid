Argumentation-based reward functions
====================================

By default, |project_name| uses numeric-based reward functions, such as
:py:class:`~smartgrid.rewards.numeric.differentiated.adaptability.AdaptabilityThree`.

From v1.2.0 onwards, you may also use *argumentation*-based reward functions,
which use an argumentation structure rather than a pure mathematical function.
Argumentation uses arguments and attacks to define how to "judge" the behaviour
of the learning agents.

You can use argumentation:

- with the existing reward functions specified in |project_name|, which
  correspond to the 4 moral values that were defined in :doc:`use_case`:
  *Affordability*, *Environmental Sustainability*, *Inclusiveness*, and
  *Supply Security*;
- to create your own reward functions, by directly using the `AJAR`_ library
  to define argumentation graphs that correspond to your desired moral values.

Using the existing argumentation reward functions
-------------------------------------------------

You can import these reward functions from the :py:mod:`smartgrid.rewards.argumentation`
package; accessing this packages *requires* the `AJAR`_ library, which you can
install with ``pip install git+https://github.com/ethicsai/ajar.git@v1.0.0``.
Trying to import anything from this package without having `AJAR`_ will raise
an error.

The 4 reward functions can be imported as such:

.. code-block:: Python

   from smartgrid.rewards.argumentation import (
        Affordability,
        EnvironmentalSustainability,
        Inclusiveness,
        SupplySecurity
   )

Then, you can create a new instance of the SmartGrid environment, exactly
as when creating a :doc:`custom_scenario`:

.. code-block:: Python

   # 1. Load agents' profiles
   converter = DataOpenEIConversion()
   converter.load('Household',
                  find_profile_data('openei', 'profile_residential_annually.npz'),
                  comfort.flexible_comfort_profile)
   # 2. Create agents
   agents = []
   for i in range(20):
       agents.append(
           Agent(f'Household{i+1}', converter.profiles['Household'])
       )
   # 3. Create world
   generator = RandomEnergyGenerator()
   world = World(agents, generator)
   # 4. Choose reward functions
   rewards = [
       Affordability(),
       EnvironmentalSustainability(),
       Inclusiveness(),
       SupplySecurity(),
   ]
   # 5. Create env
   simulator = SmartGrid(
       world,
       rewards,
       max_step,
       ObservationManager()
   )
   # 6. (Optional) Wrap the env to return scalar rewards (average)
   simulator = WeightedSumRewardAggregator(simulator)

Step *4* is the most important here: this is where you define the
argumentation-based reward functions. We have specified all 4 in this example,
but you may select only some of them, or a single one, as you desire: they work
independently.

Because we have specified here 4 different moral values, you may also use a
wrapper (:py:class:`~smartgrid.wrappers.reward_aggregator.WeightedSumRewardAggregator`)
that returns the average of the various rewards as a scalar reward (for
single-objective reinforcement learning). If you want to use a multi-objective
reinforcement learning algorithm, you can skip step 6.

The environment will work exactly as when using numeric-based reward functions;
use the standard interaction loop to make your agents receive observations and
make decisions based on them.

Writing custom argumentation reward functions
---------------------------------------------

You can also use the `AJAR`_ library to create your own argumentation-based
reward functions. This requires 3 steps:

1. Creating the argumentation graph (:py:class:`~ajar.afdm.AFDM`), with arguments
   attacks.
2. Creating the :py:class:`~ajar.judging_agent.JudgingAgent`, which will perform
   the actual judgment, i.e., transforming the symbolic arguments into a scalar
   reward.
3. Creating the :py:class:`~smartgrid.rewards.reward.Reward`, which will wrap
   the judging agent into something usable by |project_name|.

The most important step here is the 1st one, which will truly define how the
reward function works, which behaviours it will encourage, etc.

Creating the argumentation graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The argumentation graph is created by instantiating an :py:class:`~ajar.afdm.AFDM`
and adding :py:class:`~ajar.argument.Argument`\s to it:

.. code-block:: Python

   from ajar import AFDM, Argument

   afdm = AFDM()
   decision = 'moral'
   afdm.add_argument(Argument(
       "The argument identifier here",
       "The (longer) argument description here",
       lambda s: s['some_variable'] > 3,  # The activation function
       supports=[decision]
   ))

The first parameter should be a short identifier that represents your
argument; the second one (optional) can be a longer text to help describe the
argument.

The third one (optional) is the activation function, which determines when the
argument should be considered active. In the Smart Grid use-case, we can for
example have an argument "The agent has a comfort greater than 90%", for which
the activation function will be ``s['comfort'] > 0.9``. The object ``s`` here
represents the situation to be judged. By default, in |project_name|, we provide
the :py:func:`~smartgrid.rewards.argumentation.situation.parse_situation` helper
function that will return a somewhat symbolic representation of the current
environment state and the learning agent's action.

Finally, you may set whether the argument supports or counters the ``moral``
decision. If the argument supports it (``supports=[decision]``), it means that
the argument argues the learning agent performed well with respect to this moral
value; if it counters it (``counters=[decision]``), it means the argument argues
the learning performed badly with respect to this moral value. You may also
specify neither of them, which means the argument is neutral.

After creating several arguments, you can also add attacks by specifying either
the argument name or a reference to the argument itself:

.. code-block:: Python

   afdm.add_argument(Argument(
       "other_argument"
   ))
   afdm.add_attack_relationship("The argument identifier here", "other_argument")

The attack here means the ``"The argument identifier here"`` (our first argument)
attacks the ``"other_argument"``. If the first argument is alive in a given
situation, the attacked argument must be defended by another to stay alive.

You may create as many arguments and attacks as you want. You can use
`Argumentation Reward Designer`_ for a visual interface that produces Python
code compatible with `AJAR`_.

Creating the Judging agent
^^^^^^^^^^^^^^^^^^^^^^^^^^

The next step is to create a :py:class:`~ajar.judging_agent.JudgingAgent` that
will perform the judgment. An :py:class:`~ajar.afdm.AFDM` simply holds the
argumentation graph, and can determine arguments that are acceptable in a
given situation. However, the judgment itself, which returns a scalar reward
from a set of acceptable arguments, is done by Judging agents. In particular,
they are responsible for choosing how to compute this reward; this will often
boil down to comparing the number of acceptable "supporting" arguments, and
acceptable "countering" arguments. The :py:mod:`~ajar.judgment` module offers
several such methods.

.. code-block:: Python

   from ajar import JudgingAgent, judgment

   judge = JudgingAgent("Your moral value name here", afdm, judgment.j_simple)

The first argument is the name of the moral value you want this agent to
represent, for example ``"equity"``. The second argument is the
:py:class:`~ajar.afdm.AFDM` we defined previously. Finally, the third argument
is the judgment function mentioned just above.

This agent can already be used to judge a situation, by using its
:py:meth:`~ajar.judging_agent.JudgingAgent.judge` method, such as:
``judge.judge(situation={}, decision=decision)``. However, to better work with
|project_name|, we must now wrap it in a :py:class:`~smartgrid.rewards.reward.Reward`.

Creating a Reward
^^^^^^^^^^^^^^^^^

To bridge the judging agents with |project_name|, create a class that derives
from :py:class`~smartgrid.rewards.reward.Reward`, and which overrides its
:py:meth:`~smartgrid.rewards.Reward.calculate` method to return the reward
in a given situation.

.. code-block:: Python

   from smartgrid.rewards import Reward
   from smartgrid.rewards.argumentation.situation import parse_situation

   class YourRewardNameHere(Reward):
       def __init__(self):
           super().__init__()
           self.judge = judge

       def calculate(self, world, agent):
           situation = parse_situation(world, agent)
           reward = self.judge.judge(situation, decision='moral')
           return reward

You may then use this class when instantiating a :py:class:`~smartgrid.env.SmartGrid`.
The ``judge`` refers to the variable defined above; note that the ``decision``
when judging must be the same as when defining the arguments!

In the existing argumentation-based reward functions, we encapsulate the AFDM
creation in a private ``_create_afdm()`` method in each of the Rewards classes,
and we use ``decision`` as a class attribute so that both arguments creation
and judgment can rely on the same value. This is however not mandatory: as long
as the Reward has access to a judging agent to perform the judgment, it will
work.

.. _AJAR: https://github.com/ethicsai/ajar/
.. _Argumentation Reward Designer: https://ethicsai.github.io/argumentation-reward-designer/

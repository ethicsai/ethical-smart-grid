Agents profiles
===============

The :py:class:`~smartgrid.agents.profile.agent_profile.AgentProfile` contains a
few common characteristics of Agents. It represents somewhat a "type" or "kind"
of Agents, such as *Household*, *Office*, or *School*.

Creating new Agent Profiles thus allows creating new kinds of Agents. In
particular, the following characteristics are part of Profiles:

- The *need function*, which determines the energy needed by an agent at each
  time step.
- The *production function*, which determines the energy locally produced by an
  agent at each time step.
- The *comfort function*, which determines the agent's comfort, based on its
  energy need and energy consumed during a given time step.
- The *max storage*, which is the capacity of the agent's personal battery.

In this simulator, the *AgentProfiles* are created by the
:py:class:`~smartgrid.agents.data_conversion.DataConversion`.
The *data conversion* classes are responsible for loading agent profiles from
raw data files (i.e., datasets).
These data may be hard-coded or come from realistic datasets, using various
formats. Thus, to allow any type of data file to be used, the *data conversion*
classes represent some kind of bridging interface: they are specialized to a
given type of data files (e.g, the ``data/openai`` files) and create
:py:class:`~smartgrid.agents.profile.agent_profile.AgentProfile`\ s
that are agnostic to the data files, and provide the common characteristics to
the :py:class:`~smartgrid.agents.agent.Agent`\ s.

.. _extend_data_conversion:

Data conversion
---------------

New *data conversion* classes can be created to support new datasets, using
different file formats.

To illustrate this, assume that we want to load the needs and productions from
CSV files. The principal method of a
:py:class:`~smartgrid.agents.data_conversion.DataConversion` is
:py:meth:`~smartgrid.agents.data_conversion.DataConversion.load`.

Assuming a CSV file in the following form:

.. code-block::

    Step,Need,Production
    0,100,50
    1,200,30
    2,150,70

The following code loads such a CSV into a usable agent profile:

.. code-block:: Python

    from smartgrid.agents import DataConversion
    from smartgrid.agents.profile import comfort, AgentProfile, NeedProfile, ProductionProfile

    import numpy as np
    import csv

    class CSVDataConversion(DataConversion):

        def load(self, name, data_path):

            needs = []  # Amount of need each step
            productions = []  # Amount of production each step
            with open(data_path, 'r') as file:
                reader = csv.DictReader(file, fieldnames=['Step', 'Need', 'Production'])
                for row in reader:
                    needs.append(int(row['Need']))
                    productions.append(int(row['Production']))

            # The default `NeedProfile` simply returns the need corresponding
            # to the current step (cycles over the array of given needs).
            need_profile = NeedProfile(np.asarray(needs))
            # The default `ProductionProfile` works similarly.
            production_profile = ProductionProfile(np.asarray(productions))

            # The minimum value allowed for action parameters (e.g., consuming).
            low = 0
            # The maximum value allowed for action parameters. Recommended to
            # have more than the maximum need so that agents are allowed to
            # consume as much as they need (even if it might not be desirable
            # at a given step, due to ethical considerations).
            high = max(needs) + 100

            # The agents' personal storage capacity.
            max_storage = 120

            # The agents' comfort function.
            comfort_fn = comfort.neutral_comfort_profile

            # Create the Agent Profile.
            profile = AgentProfile(
                name=name,
                action_space_low=low,
                action_space_high=high,
                max_storage=max_storage,
                need_profile=need_profile,
                production_profile=production_profile,
                action_dim=len(Action._fields),
                comfort_fn=comfort_fn
            )

            # The profile must be registered in the `profiles` dict to be
            # reused later.
            self.profiles[name] = profile

This ``CSVDataConversion`` can then be used as follows:

.. code-block:: Python

    from smartgrid.agents import Agent

    converter = CSVDataConversion()
    converter.load('MyCustomProfile', '/path/to/the/data.csv')

    my_custom_profile = converter.profiles['MyCustomProfile']

    agent = Agent('MyAgent1', my_custom_profile)

Note that, in this example, as the CSV file contains only data for the needs
and productions, other values (e.g., :py:attr:`~.AgentProfile.max_storage`) are
hard-coded. The :py:meth:`~.DataConversion.load` method also accepts any
additional keyworded parameter to specify these values externally instead, for
example:

.. code-block:: Python

    class CSVDataConversion(DataConversion):

        def load(self, name, data_path, max_storage=None):
            # Same code as above, except for `max_storage = 120` (...).
            if max_storage is None:
                max_storage = 120
            # Create the Agent Profile.
            profile = AgentProfile(
                name=name,
                action_space_low=low,
                action_space_high=high,
                max_storage=max_storage,
                need_profile=need_profile,
                production_profile=production_profile,
                action_dim=len(Action._fields),
                comfort_fn=comfort_fn
            )
            # The profile must be registered in the `profiles` dict to be
            # reused later.
            self.profiles[name] = profile

To further customize the resulting agent profile, new classes can also be
created for the :py:class:`~smartgrid.agents.profile.need.NeedProfile`
and :py:class:`~smartgrid.agents.profile.production.ProductionProfile`.
New :py:mod:`comfort functions <smartgrid.agents.profile.comfort>` can also
be implemented.

.. _extend_need_profile:

Need profile
------------

The *need function* is encapsulated in the :py:class:`~smartgrid.agents.profile.need.NeedProfile`
class. This class contains an array of values (the *needs*), and returns for
each time step the corresponding value in the array (cycling over the array if
necessary). It is thus best suited for using realistic needs coming from datasets.

Let us assume that we want a similar profile, but adding a +/- 5% random
noise on the needs at each time step, to create more diversity and variety
among agents:

.. code-block:: Python

    from smartgrid.agents.profile import NeedProfile

    import numpy as np

    class NoisedNeedProfile(NeedProfile):

        def __init__(self, need_per_hour, noise=0.05):
            super().__init__(self, need_per_hour)
            self.noise = noise

        def compute(self, step=0):
            # The "basic" need (based on the array of data).
            step %= len(self.need_per_hour)
            need = self.need_per_hour[step]
            # Compute the bounds (+/- noise%).
            min_need = int(need - self.noise * need)
            max_need = int(need + self.noise * need)
            # Return a random amount within the bounds.
            return np.random.randint(min_need, max_need)

    # Let us test the need profile now.
    # Assume here that `data` is your dataset.
    # You may load it from a CSV, or a binary file, e.g., NPZ.
    data = [100, 200, 300]
    noise = 0.05
    need_profile = NoisedNeedProfile(data, noise)

    # Assert that the need at the first step is indeed in [95, 105]
    assert 100 * 0.95 <= need.compute(step=0) <= 100 * 1.05
    # More generally, for any step t:
    for t in range(100):
        min_bound = data[t % len(data)] * (1.0 - noise)
        max_bound = data[t % len(data)] * (1.0 + noise)
        assert min_bound <= need.compute(t) <= max_bound

For a more complex example, for example using a stochastic function instead of
relying purely on a dataset, you may ignore the :py:attr:`~smartgrid.agents.profile.need.NeedProfile.need_per_hour`
array, but an extra attention must be paid to the
:py:attr:`~smartgrid.agents.profile.need.NeedProfile.max_energy_needed` attribute,
which can no longer be computed automatically by the base class. See the
following block code for an example:

.. code-block:: Python

    import numpy as np
    from smartgrid.agents import NeedProfile

    class RandomNeedProfile(NeedProfile):

        def __init__(self, lower, upper):
            # Note that you should not use `super().__init__()` here
            # because we will not use the parameters in NeedProfile.
            self.lower = lower
            self.upper = upper
            # Setting the `max_energy_needed` is very important!
            self.max_energy_needed = upper

        def compute(self, step=0):
            # Return a random amount between `lower` and `upper`
            return np.random.randint(self.lower, self.upper)

    need = RandomNeedProfile(100, 1000)
    # Test for some steps that the bounds are indeed respected
    for step in range(10):
        assert 100 <= need.compute(step) <= 1000

Setting :py:attr:`~smartgrid.agents.profile.need.NeedProfile.max_energy_needed`
is crucial for the :py:class:`~smartgrid.util.available_energy.EnergyGenerator`
in particular.

These classes can then be used in your custom ``DataConversion`` when
instantiating an agent profile.

.. _extend_production_profile:

Production profile
------------------

The *production function* is encapsulated in the
:py:class:`~smartgrid.agents.profile.production.ProductionProfile` class. This
class contains an array of values (the *productions*), and returns for each time
step the corresponding value in the array. It is thus best suited for using
realistic productions coming from datasets.

The default ``ProductionProfile`` behaves very similarly to the need profile.
Again, let us assume we want to add a small random noise:

.. code-block:: Python

    from smartgrid.agents.profile import ProductionProfile

    import random

    class NoisedProductionProfile(ProductionProfile):

        def __init__(self, production_per_hour, noise=0.05):
            super().__init__(self, production_per_hour)
            self.noise = noise

        def compute(self, step=0):
            step %= len(self.production_per_hour)
            production = self.production_per_hour[step]
            min_production = int(production - self.noise * production)
            max_production = int(production + self.noise * production)
            return random.randint(min_production, max_production)

Contrary to *need profiles*, the *production profile* does not have additional
attributes to set. The only requirement is to return a value from the
:py:meth:`~smartgrid.agents.profile.production.ProductionProfile.compute`
method. More complex use-cases, such as using a stochastic function instead
of relying purely on a dataset, are thus simplified. See the following block
code for an example:

.. code-block:: Python

    from smartgrid.agents.profile import ProductionProfile

    import random

    class RandomProductionProfile(ProductionProfile):

        def __init__(self, lower, upper):
            # Again, we do not use `super().__init__()` because we do not set
            # the same attributes as the base class.
            self.lower = lower
            self.upper = upper
            # Note that there is no additional (base-required) attribute to set.

        def compute(self, step=0):
            # Return a random amount between `lower` and `upper`
            return np.random.randint(self.lower, self.upper)

.. _extend_comfort_function:

Comfort functions
-----------------

The *comfort function* is any Python callable which takes a consumption (float)
and need (float) as inputs, and returns the comfort (float). It is used to
determine the degree of satisfaction (*comfort*) of an agent at each time step.
Agents that accept to consume less when necessary may use a comfort function
that returns "high" comforts even when the consumption is less than the need;
on the contrary, agents that cannot accept to do so, e.g., an hospital because
its consumption is too important, may instead use a comfort function that
returns "low" comforts when the consumption is less than the need.

The already implemented *comfort functions* in this simulator leverage the
Richard's curve (or generalized logistic function); you may use it for your own
functions. Please see its documentation for more details:
:py:func:`smartgrid.agents.profile.comfort.richard_curve`.

Alternatively, you may provide your custom function. We give an example of a
(very) simple linear comfort, which simply considers the ratio of consumption
over the need as the comfort. This means that, if the agent consumes 80% of
their need, it will have a comfort of 80% (or ``0.8``), if it consumes 40%,
the comfort will be ``0.4``, and so on. A special attention is paid to the
output range: to avoid undesired side effects (especially when computing equity),
it is recommended that the comfort lies within ``[0, 1]``.

.. code-block:: Python

    import numpy as np

    def linear_comfort_profile(consumption, need):
        # Simply return the ratio of consumption / need.
        # Thus, the comfort increases linearly with the consumption.
        comfort = consumption / need
        # Important! It is better to clip in [0,1]!
        # Not doing so would have undesired effects
        # (especially when computing equity).
        comfort = np.clip(comfort, 0, 1)
        return comfort


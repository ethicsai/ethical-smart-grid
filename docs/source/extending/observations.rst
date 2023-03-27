Observations
============

We recall that *observations* are split into two parts: *global* ones (that are
shared by all agents) and *local* ones (that are specific and individual to
each agent). In order to reduce the computational costs, this split also happens
in the code, such that global observations can be computed only once each step
(instead of once per agent).
Thus, extending observations can be done in several parts of the architecture:

- :py:class:`~smartgrid.observation.global_observation.GlobalObservation` is
  responsible for computing the shared observations.
- :py:class:`~smargrid.observation.local_observation.LocalObservation` is
  responsible for computing the individual observations.
- :py:class:`~smartgrid.observation.observation_manager.ObservationManager` is
  the main entrypoint for all things related to observations, and is used by the
  :py:class:`~smartgrid.environment.SmartGrid` environment.

We detail how to extend each of these parts below.

GlobalObservation
-----------------

Creating a completely new way to compute observations is easy: simply define
a new class (ideally a :py:func:`collections.namedtuple`), and implement its
:py:meth:`~.GlobalObservation.compute` class method (not instance method!), as
well as :py:meth:`~.GlobalObservation.reset`.

For example, let us create a global observation class that only contains the
``hour``.

.. code-block:: Python

    from collections import namedtuple

    fields = ['hour']

    class OnlyHourGlobalObservation(namedtuple('OnlyHourGlobalObservation', fields)):

        @classmethod
        def compute(cls, world):
            hour = (world.current_step % 24) / 24
            return cls(hour=hour)

        @classmethod
        def reset(cls):
            pass

It is a little bit trickier to retain the existing fields of the global
observation, because of the way Python handles namedtuples. For another example,
let us create new *global* observations that include the current day in addition
to the existing fields.

.. code-block:: Python

    from smartgrid.observation import GlobalObservation
    from collections import namedtuple

    # `GlobalObservation._fields` is a tuple, we cannot concatenate a list to it.
    fields = ('day',) + GlobalObservation._fields

    class GlobalObservationAndDay(namedtuple('GlobalObservationAndDay', fields)):

        @classmethod
        def compute(cls, world):
            obs = GlobalObservation.compute(world)
            # `obs` is an instance (tuple) of GlobalObservation that contains
            # all the other fields we want.
            # We need to compute `day` now.
            day = world.current_step // 24
            # Now, we need to combine `day` with the other fields. To avoid
            # potential errors in the order of arguments, we will use keyworded
            # arguments (transforming `obs` into a dict and using the `**` operator).
            existing_fields = obs._asdict()
            return cls(day=day, **existing_fields)

        @classmethod
        def reset(cls):
            GlobalObservation.reset()


LocalObservation
----------------

*Local* observations follow the same principle as *global* ones: a new class
should be created. For example, let us create a new class that computes the
difference between the agents' comfort and the average of others' comfort.

.. code-block:: Python

    from collections import namedtuple
    import numpy as np

    class ComfortDiffLocalObservation(namedtuple('ComfortDiffLocalObservation', ['comfort_diff'])):

        @classmethod
        def compute(cls, world, agent):
            self_comfort = agent.comfort
            others_comforts = [a.comfort for a in world.agents]
            others_avg_comfort = np.mean(others_comforts)
            diff = self_comfort - others_avg_comfort
            return cls(comfort_diff=diff)

        @classmethod
        def reset(cls):
            # In most cases, this method will not do anything.
            # But it is provided, to allow for more complex local observations.
            pass


ObservationManager
------------------

Finally, the new classes used for computing *global* and *local* observations
must be registered with the
:py:class:`ObservationManager <smartgrid.observation.observation_manager.ObservationManager>`
so that they are used by the environment when observations must be computed.

This class already contains attributes for global and local observations; thus,
in most cases, simply creating an instance with the correct parameters should
suffice.

For example, assuming that we want to use our ``GlobalObservationAndDay``:

.. code-block:: Python

    from smartgrid.observation import ObservationManager

    # It is important to use the **type** here, not an instance of the class!
    manager = ObservationManager(
        global_observation=GlobalObservationAndDay
    )

Both *global* and *local* observations can be overriden at the same time, by
specifying both arguments:

.. code-block:: Python

    from smartgrid.observation import ObservationManager

    manager = ObservationManager(
        local_observation=ComfortDiffLocalObservation,
        global_observation=GlobalObservationAndDay
    )

The resulting ``manager`` must be specified to the
:py:class:`SmartGrid <smartgrid.environment.SmartGrid>` environment when
instantiating it:

.. code-block:: Python

    from smartgrid import SmartGrid

    env = SmartGrid(
        world=...,    # Left out as not relevant here
        rewards=...,  # Left out as well
        obs_manager=manager
    )

Energy generators
=================

The *energy generator* is responsible for determining the amount of energy
that is available in the smart grid at each time step (i.e., a local power
plant).

Extending this class allows controlling this amount to make the simulation
easier, or more difficult, or more realistic, etc., to learning agents.

In order to allow for a large variety of *energy generators*, the
:py:meth:`~smartgrid.util.available_energy.EnergyGenerator.generate_available_energy`
method, which is responsible for determining the amount available at a given
time step, receives: 1) the current sum of needs of all agents, 2) the current
step, 3) the minimum sum of needs from all agents, and 4) the maximum sum of
needs from all agents.

These data allow, e.g., scaling the available energy to the current number of
agents through their current/min/max total need. As the available energy is
part of the observations, it is important to know the possible bounds of the
generator, in order to specify the observation space (and ultimately to be
able to scale observations to ``[0, 1]`` for easier handling by learning
algorithms). The :py:meth:`~smartgrid.util.available_energy.EnergyGenerator.available_energy_bounds`
serves this purpose. The method *could* return different bounds at each time
step, but this is not recommended, as this would make it harder to define the
observation space.

In order to create a new *energy generator*, these two methods must be
implemented. Let us assume that we would like an energy generator that returns
a realistic amount, based on real-world data, independently of the agents' needs:

.. code-block:: Python

    from smartgrid.util import EnergyGenerator

    class RealworldEnergyGenerator(EnergyGenerator):

        def __init__(self, data):
            super().__init__(self)
            self.data = data

        def generate_available_energy(self,
                                      current_need,
                                      current_step,
                                      min_need,
                                      max_need):
            # Retrieve the current step from the data (cycling over the array).
            step = current_step %= len(self.data)
            energy = self.data[step]
            return energy

        def available_energy_bounds(self,
                                    current_need,
                                    current_step,
                                    min_need,
                                    max_need):
            # As we return values from the data directly, the bounds are simply
            # the min and max values of the data.
            lower_bound = min(self.data)
            upper_bound = max(self.data)
            return lower_bound, upper_bound

Note that it is very important that the result of :py:meth:`~.EnergyGenerator.generate_available_energy`
is contained within the bounds specified by :py:meth:`~.EnergyGenerator.available_energy_bounds`;
otherwise, the observation space might be incorrect, and learning agents will
receive incoherent observations.
The bounds might not need to be extremely precise (although it would be better):
very large values such as ``0`` and ``1_000_000`` can be used if no accurate
approximation can be determined.

Once the *energy generator* is defined, it can be used when instantiating the
:py:class:`~smartgrid.world.World`:

.. code-block:: Python

    from smartgrid import World

    # Generates 100Wh at t=0, then 200Wh at t=1, 300Wh at t=2, 10Wh at t=3,
    # 100Wh at t=4, and so on...
    data = [100, 200, 300, 10]
    generator = RealworldEnergyGenerator(data)

    world = World(
        agents=...,  # Not relevant here
        energy_generator=generator
    )

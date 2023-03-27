Why extending
=============

The implemented simulator already tries to support
:doc:`customization </custom_scenario>`, e.g., by letting users control the
number of instantiated agents, the energy generator, etc. However, this control
is limited to elements that are already implemented.

This simulator was designed to be extended, in order to introduce more variety,
your own data, your moral values, and so on.
To further customize the simulator, users of this package may extend the
existing classes and provide them instead of the default ones when creating
an instance of the environment. These pages describe what elements can be
extended, what are their role in the simulator, and how they can be extended.

.. list-table:: Extension points
   :header-rows: 1

   * - Element
     - Role

   * - :doc:`Observations <observations>`
     - Describe the current state of the environment to agents. Important to extend to provide different observations to agents, change the definition of a state.

Documentation of |project_name|
===============================

This project aims to provide a (simplified) multi-agent simulator of a
**Smart Grid**, using the `Gymnasium <https://gymnasium.farama.org/>`_
(formerly OpenAI Gym) framework.

This simulator has a strong focus on **ethical considerations**: in this
environment, the learning agents must decide how to consume and distribute
energy to satisfy their own need, while taking into account the other agents.

In this regard, the simulated Smart Grid is somewhat simplified: it is an
interesting use-case raising ethical considerations, but is not developed to
the point of a realistic simulator.

See :doc:`use_case` for a description of the Smart Grid use-case implemented
in this simulator; and :doc:`usage` for a quick guide on how to use this
simulator.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction

   use_case
   usage
   custom_scenario
   adding_model

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Extending the environment

   extending/index
   extending/rewards
   extending/observations

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   generated/modules

.. autosummary::
   :toctree: _autosummary
   :recursive:

   smartgrid

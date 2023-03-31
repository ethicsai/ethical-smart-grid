The SmartGrid use-case
======================

This page describes the *Smart Grid* use-case that is implemented in this
package.

Summary
-------

A *Smart Grid* is a small neighborhood in which multiple agents, also called
*prosumers* as they both produce and consume energy, interact and exchange
energy.
The goal for the agents is to decide how to distribute energy: how much should
they consume, buy, give, etc., while taking into account several ethical
considerations (moral values).

The smart grid is assumed to contain a local source of energy, such as a
hydraulic power plant, a windmill farm, etc., that is shared by all prosumers.
Each step, it generates a certain amount of energy that is freely accessible to
all agents.

The smart grid is also linked to the national grid, which is considered (for
simplification purposes) an unlimited supply of energy. Agents may buy energy
from it, or sell energy from their personal battery to the national grid.

.. image:: /images/smartgrid.drawio.png

Agents
------

Prosumer agents represent a building (or a portion of a building), such as
a *Household*, an *Office*, or a *School*. They serve as proxies for the
buildings' inhabitants, as they decide how to distribute energy.

To do so, they receive *Observations* that describe the current state of the
smart grid, and must take *Actions* that determine how much they consume, buy,
etc.

They also each possess a (small) personal storage ("battery"), whose capacity
depends on the agent's profile (a school has a larger battery than a household).
This personal storage is slightly filled each time step by their personal
production of energy (e.g., a small solar panel).

Actions
-------

Actions are defined as continuous and multi-dimensional vectors of parameters.
All agents must take an action at each time step, in order to update the
current state of the smart grid.
Each parameter represents a quantity of energy in Wh:

- ``grid_consumption``: how much energy the agent should consume from the smart
  grid's local power plant.
- ``storage_consumption``: how much energy the agent should consume from its
  own personal storage.
- ``store_energy``: how much energy the agent should take from the local power
  plant to store in its personal battery.
- ``give_energy``: how much energy the agent should give from its personal
  battery to the local power plant.
- ``buy_energy``: how much energy the agent should buy from the national grid
  to store in its personal battery.
- ``sell_energy``: how much energy the agent should sell from its personal
  battery to the national grid.

In other words, the action parameters define how to distribute, or transfer,
energy between the smart grid, the national grid, and the agent's storage
and consumption.

.. image:: /images/energy_transfers.drawio.png

Observations
------------

Observations are defined as continuous and multi-dimensional vectors that
describe the current state of the environment.
Privacy is an important aspect of this use-case -- we would not want our
neighbours to know how much we consumed! --, thus, observations are split
into *global* (the same values shared by all agents at a given time step)
and *local* (personal to each agent) observations.

- Global:

  * ``hour``: The current hour of the environment.
  * ``available_energy``: The current amount of energy available from the
    smart grid's local power plant (i.e., freely accessible to agents).
  * ``equity``: The measure of equity between agents' comforts at the previous
    time step, typically using a statistical metric such as
    `Hoover index <https://en.wikipedia.org/wiki/Hoover_index>`_.
  * ``energy_loss``: The proportion of energy that was available but not used
    at the previous time step.
  * ``autonomy``: The degree to which agents avoided using the national grid.
  * ``exclusion``: The proportion of agents that had a comfort lower than
    half the median of comforts.
  * ``well_being``: The median of comforts.
  * ``over_consumption``: The amount of energy that was consumed by agents,
    but not initially available in the smart grid's local power plant. We
    assume that the smart grid is able to compensate by automatically taking
    energy from the national grid.

- Local:

  * ``personal_storage``: The current proportion of energy available in the
    agent's personal battery, with respect to its capacity.
  * ``comfort``: The comfort the agent had at the previous time step, by
    consuming energy.
  * ``payoff``: The agent's current sum of benefits and losses from selling
    and buying energy to/from the national grid.


Moral values
------------

This environment was created to learn "ethical behaviours", i.e., behaviours
aligned with moral values; several values are therefore considered.

These values have been found in the scientific literature ([deWildt]_,
[Milchram]_, [Boijmans]_); however, their original definition was focused on the
policymakers' point of view, e.g., "the construction of a smart grid must allow
prosumers to do X". Instead, as the moral values must be respected by the
prosumer agents, and as these agents act as proxies for inhabitants of the
smart grid, the moral values were adapted to take their point of view.

Security of Supply
    An action that allows a prosumer to improve its comfort is moral.

Affordability
    An action that makes a prosumer pay too much money is immoral.

Inclusiveness
    An action that improves the equity of comforts between all prosumers is
    moral.

Environmental Sustainability
    An action that prevents transactions (buying or selling energy) with the
    national grid is moral.


Note that these moral values may be in conflict in some situations. For example,
let us assume that the quantity of available energy is limited, and that, at
some time step, there is not enough for all agents to completely satisfy
their comfort (i.e., consume at least as much as their need).
If an agent consumes too much, they risk preventing another agent from
satisfying its own comfort, thus betraying the *inclusiveness* moral value.
Instead, it may buy energy to compensate, but this would betray the
*environmental sustainability* value, and potentially the *affordability* as
well (if the agent does not have enough money). Finally, the agent may simply
choose to consume less, thus reducing its comfort and betraying the
*security of supply* value.

These (potential) conflicts between values make this Smart Grid environment
a suitable playground for learning "ethical behaviours".


.. [deWildt] Wildt, T. E. de, E. J. L. Chappin, G. van de Kaa, P. M. Herder, and I. R. van de Poel. “Conflicting Values in the Smart Electricity Grid a Comprehensive Overview.” Renewable and Sustainable Energy Reviews 111 (September 1, 2019): 184–96. https://doi.org/10.1016/j.rser.2019.05.005.


.. [Milchram] Milchram, Christine, Geerten Van de Kaa, Neelke Doorn, and Rolf Künneke. “Moral Values as Factors for Social Acceptance of Smart Grid Technologies.” Sustainability 10, no. 8 (August 2018): 2703. https://doi.org/10.3390/su10082703.


.. [Boijmans] Boijmans, Anne R. “The Acceptability of Decentralized Energy Systems: Identifying Value Conflicts Through Simulations Of Decentralized Energy Systems For City Districts.” Master Thesis, Delft University of Technology, 2019. https://pdfs.semanticscholar.org/7c5b/3311776ec794356793eabfda718236e4738d.pdf.


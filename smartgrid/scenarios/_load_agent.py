"""
This module offers helper functions to instantiate Agents from profiles.

Agent profiles are loaded from the `.npz` files in the `data/` folder.
Npz files are archives that contain multiple NumPy arrays.
In this case, they must contain:

action_space_low
    an array of lower bounds for the agents' actions
    (e.g., the 1st value is the lower bound of the 1st action dimension, etc.)
action_space_high
    an array of upper bounds for agents' actions
    (must have the same shape as `action_space_low`)
max_storage
    a single value array, which is the maximum battery capacity
    (note: we would like `max_storage` to be a scalar value... but npz files
    can only contain arrays, so we make it an array of a single value)
need_per_hour
    an array of "needs" for each hour of a year
    (needs represent the amount the agent wants to consume)
production_per_hour
    an array of "production" for each hour of a year
    (the amount of energy produced by the agent's personal solar panel)

An example of how to create such files::

    np.savez(filepath,
             action_space_low=[0, 0, 0, 0, 0, 0],
             action_space_high=[100, 100, 100, 100, 100, 100],
             max_storage=[500],
             need_per_hour=[123, 456, 789, ...],
             production_per_hour=[321, 654, 987, ...])

(Alternatively, `np.savez_compressed` can be used to reduce the file size)

This module offers the following functions:

_load_profile
    automates parsing and validating profiles from `.npz` files
_compute_need
    a helper function, that returns a need from a time step,
    and an array of needs per hour. However, the Agent's `compute_need` signature
    takes only the time step as input, so we apply a partial on `_compute_need`
    to bind the `need_per_hour` argument to the desired array when we instantiate
    Agents.
_compute_production
    similar to `_compute_need`, but for the production.

"""

from functools import partial

from smartgrid.agent import Agent
from smartgrid.util import comforts
from smartgrid.util import Profile


def _compute_need(step, need_per_hour):
    step = step % len(need_per_hour)
    return need_per_hour[step]


def _compute_production(step, production_per_hour):
    step = step % len(production_per_hour)
    return production_per_hour[step]


def _create_agent(name, comfort_fn, profile_filename):
    profile = Profile.load(profile_filename)

    action_space = profile.action_space
    max_storage = profile.max_storage
    compute_need = partial(_compute_need,
                           need_per_hour=profile.need_per_hour)
    compute_production = partial(_compute_production,
                                 production_per_hour=profile.production_per_hour)
    max_energy_needed = max(profile.need_per_hour)

    agent = Agent(name,
                  action_space,
                  max_storage,
                  max_energy_needed,
                  comfort_fn,
                  compute_need,
                  compute_production,
                  )
    return agent


def create_household(i):
    profile_filename = 'test_profile.npz'
    comfort_fn = comforts.comfort_flexible
    name = f'Household[Flexible] {i}'
    return _create_agent(name, comfort_fn, profile_filename)

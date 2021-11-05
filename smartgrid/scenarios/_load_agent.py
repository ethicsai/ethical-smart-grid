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
from pathlib import Path

import numpy as np
from gym import spaces

from smartgrid.agent import Agent
from smartgrid.util import comforts


def _load_profile(profile_path):
    """Parse and validate an agent profile from a `.npz` file.

    If the profile does not contain the expected data, or they do not
    meet their constraints (specifically, in terms of shapes), this
    will raise exceptions.
    See this module's documentation for the expected data format.
    """
    # FIXME: replace the `raise()` calls by real exceptions
    # First, create the "static" map to memorize profiles
    # (this allows avoiding to reload and reparse the file between
    # multiple invocations)
    if not hasattr(_load_profile, 'memorized'):
        _load_profile.memorized = {}
    # If the profile was already parsed, return it immediately
    if profile_path in _load_profile.memorized.keys():
        return _load_profile.memorized[profile_path]
    # Else, load the profile from the `npz` file (we will then memorize it)
    with np.load(profile_path) as profile:
        # Check if profile is correctly formatted
        for key in ['action_space_low', 'action_space_high', 'max_storage',
                    'need_per_hour', 'production_per_hour']:
            if key not in profile.files:
                raise('Profile incorrectly formatted! Missing element: ' + key)
        # Parse the profile
        action_space_low = profile['action_space_low']
        action_space_high = profile['action_space_high']
        if action_space_low.shape != action_space_high.shape:
            raise('low and high bounds of action space do not have same shape!')
        action_space = spaces.Box(
            low=action_space_low,
            high=action_space_high,
            dtype=np.int
        )
        # .npz file only store arrays, we want `max_storage` as a single value
        max_storage = profile['max_storage'][0]
        need_per_hour = profile['need_per_hour']
        production_per_hour = profile['production_per_hour']
        if need_per_hour.shape != production_per_hour.shape:
            raise('Need and production per hour do not have the same shape!')
        # Memorize the parsed profile
        parsed_profile = {
            'action_space': action_space,
            'max_storage': max_storage,
            'need_per_hour': need_per_hour,
            'production_per_hour': production_per_hour
        }
        _load_profile.memorized[profile_path] = parsed_profile
        return parsed_profile


def _compute_need(step, need_per_hour):
    step = step % len(need_per_hour)
    return need_per_hour[step]


def _compute_production(step, production_per_hour):
    step = step % len(production_per_hour)
    return production_per_hour[step]


def _create_agent(name, comfort_fn, profile_filepath):
    profile = _load_profile(profile_filepath)

    action_space = profile['action_space']
    max_storage = profile['max_storage']
    compute_need = partial(_compute_need,
                           need_per_hour=profile['need_per_hour'])
    compute_production = partial(_compute_production,
                                 production_per_hour=profile['production_per_hour'])

    agent = Agent(name,
                  action_space,
                  max_storage,
                  comfort_fn,
                  compute_need,
                  compute_production,
                  )
    return agent


def create_household(i):
    profile_filepath = Path(__file__) / '../../../data' / 'test_profile.npz'
    profile_filepath = profile_filepath.resolve()
    comfort_fn = comforts.comfort_flexible
    name = f'Household[Flexible] {i}'
    return _create_agent(name, comfort_fn, profile_filepath)

from functools import partial
from pathlib import Path

import numpy as np
from gym import spaces

from smartgrid.agent import Agent
from smartgrid.util import comforts


def _load_profile(profile_path):
    # FIXME: replace the `raise()` calls by real exceptions
    # FIXME: memorize profiles
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
        # Return the parsed profile
        return {
            'action_space': action_space,
            'max_storage': max_storage,
            'need_per_hour': need_per_hour,
            'production_per_hour': production_per_hour
        }


def _compute_need(step, need_per_hour):
    step = step % len(need_per_hour)
    return need_per_hour[step]


def _compute_production(step, production_per_hour):
    step = step % len(production_per_hour)
    return production_per_hour[step]


def create_household(i):
    profile_filepath = Path(__file__) / '../../../data' / 'test_profile.npz'
    profile_filepath = profile_filepath.resolve()
    profile = _load_profile(profile_filepath)

    action_space = profile['action_space']
    max_storage = profile['max_storage']
    comfort_fn = comforts.comfort_flexible
    compute_need = partial(_compute_need,
                           need_per_hour=profile['need_per_hour'])
    compute_production = partial(_compute_production,
                                 production_per_hour=profile['production_per_hour'])
    name = f'Household[Flexible] {i}'

    agent = Agent(name,
                  action_space,
                  max_storage,
                  comfort_fn,
                  compute_need,
                  compute_production,
                  )
    return agent

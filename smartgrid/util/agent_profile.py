"""
This module makes it easier to save and load agents' Profiles.

An agent's Profile contains data necessary to instantiate an Agent,
e.g., the agent's production and need for each hour.
Usually, these data are generated from a dataset, which makes it useful
to save them as intermediate files, and to reload them when starting a
simulation.

This module stores Profiles using NumPy's `.npz` format, in the `data/`
folder, at the root of this project.
Npz files are archives (similar to ZIP) that contain multiple NumPy arrays.
In this case, they must contain the following data:

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

Profiles can be created by constructing a new instance, saved with the
`Profile.save(filename)` instance method, and loaded with the
`Profile.load(filename)` class method.

This class uses NumPy arrays for most of its attributes (except `max_storage`,
which is a scalar value), but regular Python lists can be given to the
constructor: they will be transformed to ndarrays.

See the following code for an example of how to use this class::

    profile1 = Profile(
        action_space_low=[0, 0, 0, 0, 0, 0],
        action_space_high=[100, 100, 100, 100, 100, 100],
        max_storage=500,
        need_per_hour=[128, 512, 477],
        production_per_hour=[98, 201, 113]
    )
    profile1.save('profile1.npz')

    profile2.load('profile2.npz')
    assert profile2.need_per_hour.shape == profile2.production_per_hour
"""

from pathlib import Path

import numpy as np
from gym import spaces


class Profile(object):
    # Class-scoped attributes
    # Path to the `data/` folder
    _data_path = Path(__file__).parent / '../../data'
    # Memorize already-loaded profiles to avoid unnecessary computations
    _memorized_profiles = {}

    def __init__(self,
                 action_space_low,
                 action_space_high,
                 max_storage,
                 need_per_hour,
                 production_per_hour):

        self.action_space_low = np.asarray(action_space_low)
        self.action_space_high = np.asarray(action_space_high)

        if self.action_space_low.shape != self.action_space_high.shape:
            raise Exception('action_space_low and action_space_high must '
                            'have the same shape! Found '
                            f'{action_space_low.shape} and '
                            f'{action_space_high.shape}')

        self.need_per_hour = np.asarray(need_per_hour)
        self.production_per_hour = np.asarray(production_per_hour)

        if self.need_per_hour.shape != self.production_per_hour.shape:
            raise Exception('need_per_hour and production_per_hour must '
                            'have the same shape! Found '
                            f'{need_per_hour.shape} and '
                            f'{production_per_hour.shape}')

        self.max_storage = max_storage

        # For easier access, we pre-compute the Action Space from low and high
        self.action_space = spaces.Box(
            low=action_space_low,
            high=action_space_high,
            dtype=np.int
        )

    def save(self, filename):
        """Save a valid Profile to a `.npz` file in the `data/` folder.

        This method ensures the data is formatted in a way understandable
        by the `Profile.load(filename)` method.

        :param filename: The name of the file in which the Profile will be
            saved. It *must* end with the `.npz` extension, and will be
            resolved relatively to the `data/` folder at the project root.
        :type filename: str
        """
        filepath = Profile.path_to(filename)
        np.savez_compressed(
            filepath,
            action_space_low=self.action_space_low,
            action_space_high=self.action_space_high,
            max_storage=np.array(self.max_storage),  # Note: 0d array
            need_per_hour=self.need_per_hour,
            production_per_hour=self.production_per_hour
        )

    @classmethod
    def load(cls, filename):
        """Parse and validate an agent's Profile from a `.npz` file.

        If the profile does not contain the expected data, or if they
        do not meet their constraints (especially in terms of shapes),
        this method will raise exceptions.

        :param filename: The name of the file to load. It will be resolved
            relatively to the `data/` folder in the project root, similarly
            to the `Profile.save(filename)` method.
        :type filename: str

        :return: A valid instance of Profile, containing the loaded data.
        :rtype: Profile

        :raises FileNotFoundError: If the file does not exist.
        :raises Exception: In case of errors in the data (missing elements,
            or elements with an incorrect shape). The exception's message
            details the problem.
        """
        if filename in cls._memorized_profiles:
            # The profile was already loaded once, return it immediately
            return cls._memorized_profiles[filename]
        # Else, we need to load the profile from the `.npz` file.
        # We will then memorize it.
        filepath = cls.path_to(filename)
        with np.load(filepath) as profile:
            # Assert that we have all necessary data
            expected_keys = ['action_space_low', 'action_space_high',
                             'max_storage', 'need_per_hour',
                             'production_per_hour']
            missing_keys = [k for k in expected_keys if k not in profile.files]
            if len(missing_keys) > 0:
                raise Exception(f'Profile {filename} incorrectly formatted! '
                                f'Missing elements: {missing_keys}')

            # .npz files only store arrays, we want `max_storage` a single value
            max_storage = profile['max_storage']
            if len(max_storage.shape) == 0:
                # If `max_storage` is a 0d array, we cannot index it directly
                # But we can use an empty tuple (i.e., a tuple of 0d)
                max_storage = max_storage[()]
            elif len(max_storage.shape) == 1:
                # A simple 1d array. Get the first (and single?) value
                max_storage = max_storage[0]
            else:
                raise Exception('max_storage should be a 0d or 1d array, '
                                f'found {max_storage.shape}')

            # Create the profile (will also check for correct shapes)
            profile = Profile(
                action_space_low=profile['action_space_low'],
                action_space_high=profile['action_space_high'],
                max_storage=max_storage,
                need_per_hour=profile['need_per_hour'],
                production_per_hour=profile['production_per_hour']
            )
        # Memorize the profile
        cls._memorized_profiles[filename] = profile
        return profile

    @classmethod
    def path_to(cls, filename):
        return str((cls._data_path / filename).resolve())

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from agents.agent import Action
from agents.profile.agent_profile import AgentProfile
from agents.profile.need import NeedProfile
from agents.profile.production import ProductionProfile


class DataConversion(ABC):
    memorized_profiles: Dict[str, AgentProfile]
    name: str
    max_step: int

    def __init__(self, name: str):
        self.name = name
        self.max_step = 0

    def __str__(self):
        return self.name

    @abstractmethod
    def load(self, name: str, config: tuple) -> None:
        pass

    @property
    def profiles(self) -> Dict[str, AgentProfile]:
        return self.memorized_profiles


class DataOpenEIConversion(DataConversion):
    # Assert that we have all necessary data
    expected_keys = ['needs', 'action_limit',
                     'max_storage']

    def __init__(self):
        super().__init__("OpenEI")
        self.memorized_profiles = {}

    def load(self, name: str, config: tuple) -> None:
        # TODO change commentary
        """
        This module offers helper functions to instantiate Agents profiles.

        Data are loaded from the `.npz` files in the `data/` folder.
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
            (needs to represent the amount the agent wants to consume)
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
        check = np.load(config[0])
        missing_keys = [k for k in self.expected_keys if k not in check.files]
        if len(missing_keys) > 0:
            raise Exception(f'Profile {config} incorrectly formatted! '
                            f'Missing elements: {missing_keys}')

        if len(check['max_storage'].shape) > 1:
            raise Exception('max_storage should be a 0d or 1d array, '
                            f'found {check["max_storage"].shape}')

        # .npz files only store arrays, we want `max_storage` a single value
        max_storage = check['max_storage']
        if len(max_storage.shape) == 0:
            # If `max_storage` is a 0d array, we cannot index it directly
            # But we can use an empty tuple (i.e., a tuple of 0d)
            max_storage = max_storage[()]
        elif len(check['max_storage'].shape) == 1:
            # A simple 1d array. Get the first (and single?) value
            max_storage = max_storage[0]

        # loading needs into profile
        need_profile = NeedProfile(np.asarray(check["needs"]))

        # generate production data from needs
        # todo put production
        productions = [0] * len(check["needs"])
        production_profile = ProductionProfile(np.asarray(productions))

        # searching low and high action
        low = np.int64(0)
        high = max(check["needs"])

        # Create the profile (will also check for correct shapes)
        profile = AgentProfile(
            name=name,
            action_space_low=low,
            action_space_high=high,
            max_storage=max_storage,
            need_profile=need_profile,
            production_profile=production_profile,
            action_dim=len(Action._fields),
            comfort_fn=config[1]
        )

        self.memorized_profiles[name] = profile

        self.max_step = len(need_profile.need_per_hour)

"""
This module is used to convert raw data into Agent Profiles.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from .agent import Action
from .profile import (AgentProfile,
                      NeedProfile,
                      ProductionProfile)


class DataConversion(ABC):
    """
    Convert raw data into usable :py:class:`.AgentProfile`\\ s.

    To improve re-usability and because they may contain important amounts of
    data (e.g., quantity of energy needed for each step), profiles are usually
    stored as data files, using some specific format.
    DataConversion classes are responsible for translating such data files
    and loading them into tangible (instantiated) profiles.

    New DataConversion classes can be created to handle different file formats
    and structures, so that the simulator itself is agnostic to the data source.
    """

    profiles: Dict[str, AgentProfile]
    """Profiles already loaded by the DataConversion, to speed up next calls."""

    def __init__(self):
        self.profiles = {}

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def load(self, name: str, data_path: str, **kwargs) -> AgentProfile:
        """
        Load a profile from a data file for further use.

        :param name: The desired profile name. This can be seen as the
            profile's ID, as the name must be used to later retrieve the
            profile from the :py:attr:`.profiles` dict.

        :param data_path: The path to the data file from which the profile
            should be loaded. This path must exist and be readable.

        :param kwargs: Additional arguments.
            These arguments can serve any purpose, depending on the
            implementation details of the DataConversion itself.

        :return: The loaded AgentProfile for direct use.
        """
        pass


class DataOpenEIConversion(DataConversion):
    """
    DataConversion specialized for data coming from the OpenEI dataset.

    Data that were extracted from the OpenEI dataset have been transformed
    as NPZ files for easier and faster loading from Python. They should all
    have the same structure:

    - ``needs``: A NumPy array describing the quantity of energy needed each step.
    - ``action_limit``: The upper bound of the agent's action.
    - ``max_storage``: The capacity of the agent's personal storage.

    Note that OpenEI-based profiles do not contain production or comfort:
    we must generate them ourselves.
    As such, the :py:meth:`.load` method requires an additional ``comfort_fn``
    argument (keyworded, e.g., ``comfort_fn=...``).
    """

    expected_keys = ['needs', 'action_limit', 'max_storage']
    """Keys that are expected in the NpzFile loaded from the data file."""

    # We have to redefine the profiles here, otherwise, Sphinx complains that
    # it does not exist, when building the documentation, although it *is*
    # in the parent class... *sigh*
    profiles: Dict[str, AgentProfile]

    def load(self, name, data_path, comfort_fn=None) -> AgentProfile:
        """
        Load a profile from an OpenEI-based data file.

        These data files can be found in the `data/openei` directory.

        :param name: The desired profile name. This can be seen as the profile's
            ID, as it will be used to later retrieve it from the
            :py:attr:`.profiles` dict.

        :param data_path: The path to the data file from which the profile
            should be loaded. This path must exist and be readable.

        :param comfort_fn: The comfort function that should be used. See
            :py:mod:`~smartgrid.agents.profile.comfort` for details on comfort
            functions.

        :return: The loaded AgentProfile for direct use.
        """

        # Load the NPZ file
        content = np.load(data_path)

        # Check that the file's structure is correct
        missing_keys = [k for k in self.expected_keys if k not in content.files]
        if len(missing_keys) > 0:
            raise Exception(f'Profile {name} in file {data_path} incorrectly '
                            f'formatted! Missing elements: {missing_keys}')

        # Parse data from the file

        # - `max_storage`
        # .npz files only store arrays, we want `max_storage` a single value
        max_storage = content['max_storage']
        max_storage = self._get_ndarray_single_value(max_storage)

        # - `needs`
        needs = np.asarray(content['needs'])
        need_profile = NeedProfile(needs)

        # - `production`
        # OpenEI does not contain data about production, so we must generate it
        # from the needs. Let it be a random amount between 0% and 10% of the
        # max_storage for each step.
        production_upper_bound = int(0.1 * max_storage)
        productions = [
            random.randint(0, production_upper_bound)
            for _ in range(len(needs))
        ]
        production_profile = ProductionProfile(np.asarray(productions))

        # - `action_limit`
        low = np.int64(0)
        high = self._get_ndarray_single_value(content['action_limit'])

        if comfort_fn is None:
            raise Exception('The comfort function `comfort_fn` must be specified!')

        # Create the profile (will also check for correct shapes)
        profile = AgentProfile(
            name=name,
            action_space_low=low,
            action_space_high=high,
            max_storage=max_storage,
            need_profile=need_profile,
            production_profile=production_profile,
            action_dim=len(Action._fields),
            comfort_fn=comfort_fn
        )

        self.profiles[name] = profile
        return profile

    def _get_ndarray_single_value(self, array: np.ndarray):
        """Internal method to get the single value of a 0d or 1d ndarray."""
        if len(array.shape) == 0:
            # If it is a 0d array, we cannot index it directly
            # But we can use an empty tuple (i.e., a tuple of 0d)
            value = array[()]
        elif len(array.shape) == 1:
            # A simple 1d array. Get the first (and single?) value
            value = array[0]
        else:
            raise Exception('The array should be a 0d or 1d ndarray, '
                            f'found {array.shape}')
        return value

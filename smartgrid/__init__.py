"""
The smartgrid package defines the Smart Grid simulator.
"""

from gymnasium.envs.registration import register

from .make_env import make_basic_smartgrid

# Packages imported to simplify usage
from .environment import SmartGrid
from .world import World

register(
    id='EthicalSmartGrid-v0',
    entry_point=make_basic_smartgrid
)

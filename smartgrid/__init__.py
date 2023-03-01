from gymnasium.envs.registration import register

from .environment import SmartGrid
from .world import World

register(
    id='SmartGrid-v0',
    entry_point='smartgrid'
)

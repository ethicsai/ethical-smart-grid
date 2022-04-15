from abc import abstractmethod

import numpy as np
from gym.vector.utils import spaces

from smartgrid.agents.profile.need import NeedProfile
from smartgrid.agents.profile.production import ProductionProfile


class AgentProfile:
    """
    A AgentProfile is a way to resume how an agent interact with the world.
    You have many profile and all data is loaded inside his metrics.

    In the basic case of the Simulator, you have:
        - ProductionProfile: The production of the electrical device of Agent
        - NeedProfile: For the energetic need of the Agent
        - ComfortProfile: For the calculation of the Comfort at a step
    """

    def __init__(self,
                 name: str,
                 need_profile: NeedProfile,
                 production_profile: ProductionProfile,
                 max_storage,
                 action_space_low,
                 action_space_high,
                 action_dim,
                 comfort_fn):
        if action_space_low.shape != action_space_high.shape:
            raise Exception('action_space_low and action_space_high must '
                            'have the same shape! Found '
                            f'{action_space_low.shape} and '
                            f'{action_space_high.shape}')

        self.name = name

        # For easier access, we pre-compute the Action Space from low and high
        self.action_space = spaces.Box(
            low=action_space_low,
            high=action_space_high,
            shape=(action_dim,),
            dtype=np.int
        )

        # create observation_space for an agent
        space_dict = {
            'personal_storage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int),
            'comfort': spaces.Box(0, 1, (1,), dtype=float),
            'payoff': spaces.Box(0, 1, (1,), dtype=float)
        }
        self.observation_space = spaces.Dict(space_dict)

        self.need_fn = need_profile
        self.production_fn = production_profile
        self.comfort_fn = comfort_fn

        self.production = None
        self.need = None
        self.max_storage = max_storage
        self.max_energy_needed = self.need_fn.max_energy_needed

    @abstractmethod
    def update(self, step: int) -> None:
        """
        Function that update all metrics for a step.
        """
        self.need = self.need_fn.compute(step)
        self.production = self.production_fn.compute(step)

    def reset(self):
        self.production = self.production_fn.compute(0)
        self.need = self.need_fn.compute(0)

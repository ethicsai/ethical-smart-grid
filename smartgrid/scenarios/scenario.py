from abc import ABC, abstractmethod
from typing import List, Type

from smartgrid.agents.agent import Agent
from smartgrid.agents.data_conversion import DataConversion, DataOpenEIConversion
from smartgrid.agents.profile.comfort import flexible_comfort_profile, neutral_comfort_profile, strict_comfort_profile
from smartgrid.aggregate_function.aggregate_function import BasicAggregateFunction, AggregateFunction
from smartgrid.environment import SmartGrid
from smartgrid.observation.global_observation import GlobalObservation
from smartgrid.observation.local_observation import LocalObservation
from smartgrid.observation.observation_manager import ObservationManager
from smartgrid.observation.observations import Observation
from smartgrid.rewards.numeric.differentiated.equity import EquityRewardOne
from smartgrid.rewards.reward import Reward
from smartgrid.util import EnergyGenerator, RandomEnergyGenerator
from smartgrid.world import World


class Scenario(ABC):
    name: str
    confort_name: str
    observation_manager: ObservationManager
    aggregate_function: Type[AggregateFunction]
    rewards: List[Reward]
    energy_generator: EnergyGenerator
    data_conversion: DataConversion
    agents: List[Agent]
    aggregate_function_name: str

    @property
    def max_step(self):
        return self.data_conversion.max_step

    def make(self):
        self._prepare()
        world = World(self.observation_manager,
                      self.agents,
                      self.rewards,
                      Observation,
                      self.max_step,
                      self.energy_generator)
        env = SmartGrid(world)

        aggregate = self.aggregate_function(env)
        self.aggregate_function_name = str(aggregate)
        return aggregate

    @abstractmethod
    def _prepare(self):
        pass


class DefaultScenario(Scenario):

    def _prepare(self):
        self.name = "default"
        config = {
            "office": ("./data/phd/profile_office_annually.npz", neutral_comfort_profile),
            "school": ("./data/phd/profile_school_annually.npz", strict_comfort_profile),
            "residential": ("./data/phd/profile_residential_annually.npz", flexible_comfort_profile)
        }
        number = {
            "office": 5,
            "school": 1,
            "residential": 50
        }
        self.energy_generator = RandomEnergyGenerator()
        self.data_conversion = DataOpenEIConversion()

        self.data_conversion.load("office", config["office"])
        self.data_conversion.load("school", config["school"])
        self.data_conversion.load("residential", config["residential"])

        self.agents = []
        for profile_name in number.keys():
            for _ in range(number[profile_name]):
                new_agent = Agent(f"{len(self.agents)}", self.data_conversion.profiles[profile_name])
                self.agents.append(new_agent)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardOne()]

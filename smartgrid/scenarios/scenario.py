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
    """
    The purpose of this class is to be generic for defining a way to work of the Simulator.
    Note that the abstract class is made for instantiate a :py:class:`.World` with parameters generate in :py:meth:`._prepare`.
    For creating a new scenario (e.g. extending), you need to inherit from the *class* and defines :py:meth:`._prepare`.
    By this way, you can make your own object with corresponding structure and data.
    """

    name: str
    """
    Name of the Scenario. Keep unicity for clear name in your experiments.
    """
    observation_manager: ObservationManager
    """
    The :py:class:`.ObservationManager` that compute all type of Observation.
    """
    aggregate_function: Type[AggregateFunction]
    """
    The :py:class:`.AggregateFunction` wrapping reward. Extend it in case off multiple rewards.
    :py:class:`.BasicAggregateFunction` used for mono-reward.
    """
    rewards: List[Reward]
    """
    A List of :py:class:`.Reward` computed by the simulator. It will be stored in :py:class:`.RewardCollection`
    In case of single reward, make a singleton list.
    """
    energy_generator: EnergyGenerator
    """
    The :py:class:`.EnergyGenerator` compute the electricity produce by the SmartGrid.
    It will be store in :py:class:`.GlobalObservation` under available_energy field.
    """
    data_conversion: DataConversion
    """
    The :py:class:`.DataConversion` is the way that your load data in :py:class:`.AgentProfile`,
    both `consuming` and `need` need to be be load into a :py:class:`.AgentProfile` type.
    Refers to the class for more information.
    """
    agents: List[Agent]
    """
    Used a :py:class:`.AgentProfile` for creating multiple :py:class:`.Agent`.
    Construct during :py:meth:`._prepare`.
    """
    aggregate_function_name: str
    """
    The name of the RewardWrapper used.
    """

    @property
    def max_step(self):
        return self.data_conversion.max_step

    def make(self):
        """
        Construct the simulator with all object initialize with :py:meth:`._prepare`.

        :return: a :py:class:`.RewardWrapper` wrapping a instance of :py:class:`.Smartgrid`
        """
        # 1. _prepare() is the way to specify the object of the simulator (for extending purpose)
        self._prepare()
        # 2. construct the World with object create during prepare phase
        world = World(self.observation_manager,
                      self.agents,
                      self.rewards,
                      Observation,
                      self.max_step,
                      self.energy_generator)
        # 3. Construct the Gym Environment
        env = SmartGrid(world)
        # 4. Wrapping reward for your model
        aggregate = self.aggregate_function(env)
        self.aggregate_function_name = str(aggregate)
        return aggregate

    @abstractmethod
    def _prepare(self):
        """
        Instantiate all the field necessary for the construction of the World and RewardWrapper including data.
        Necessary fields:
            - :py:attr:`.aggregate_function`: The type of a :py:class:`.AggregateFunction`,
                note that is not an instance.
            - :py:attr:`.energy_generator`: The generator used for computing `available_energy`
            - :py:attr:`.observation_manager`: The manager that compute and defines all type of observation.
            - :py:attr:`.agents`: list of :py:class:`.Agent` create with an :py:class:`.AgentProfile`.
            - :py:attr:`.rewards`: list of :py:class:`.Reward`.
            - :py:attr:`.data_conversion`: a :py:class:`.DataConversion` for :py:property:`.max_step`.
        :return: nothing
        """
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

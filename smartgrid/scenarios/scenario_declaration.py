from abc import ABC
from typing import Dict

from rewards.numeric.per_agent.adaptability import AdaptabilityThreePerAgent, AdaptabilityTwoPerAgent, \
    AdaptabilityOnePerAgent
from rewards.numeric.per_agent.equity import EquityRewardPerAgent
from rewards.numeric.per_agent.multi_objective_sum import MultiObjectiveSumPerAgent
from rewards.numeric.per_agent.over_consumption import OverConsumptionPerAgent
from smartgrid.agents.agent import Agent
from smartgrid.agents.data_conversion import DataOpenEIConversion
from smartgrid.agents.profile.comfort import flexible_comfort_profile, neutral_comfort_profile, strict_comfort_profile
from smartgrid.aggregate_function.aggregate_function import BasicAggregateFunction, MultiObjectiveProduct
from smartgrid.observation.global_observation import GlobalObservation
from smartgrid.observation.local_observation import LocalObservation
from smartgrid.observation.observation_manager import ObservationManager
from smartgrid.rewards.numeric.differentiated.adaptability import AdaptabilityOne, AdaptabilityTwo, AdaptabilityThree
from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.differentiated.equity import EquityRewardTwo, EquityRewardOne
from smartgrid.rewards.numeric.differentiated.multi_objective_sum import MultiObjectiveSum
from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
from smartgrid.scenarios.scenario import Scenario
from smartgrid.util import RandomEnergyGenerator


class BaseOpenEI(Scenario, ABC):
    number: Dict[str, int]

    def _default(self):
        self.second_name = None
        paths = {
            "office": ("./data/phd/profile_office_annually.npz", neutral_comfort_profile),
            "school": ("./data/phd/profile_school_annually.npz", strict_comfort_profile),
            "residential": ("./data/phd/profile_residential_annually.npz", flexible_comfort_profile)
        }
        self.data_conversion = DataOpenEIConversion()

        self.data_conversion.load("office", paths["office"])
        self.data_conversion.load("school", paths["school"])
        self.data_conversion.load("residential", paths["residential"])

        self.agents = []
        for profile_name in self.number.keys():
            for _ in range(self.number[profile_name]):
                new_agent = Agent(f"{len(self.agents)}", self.data_conversion.profiles[profile_name])
                self.agents.append(new_agent)

        cumulative_need = [sum([a.profile.need_fn.need_per_hour[i] for a in self.agents])
                           for i in range(self.data_conversion.max_step)]
        self.lower_cumulated = min(cumulative_need)
        self.upper_cumulated = max(cumulative_need)


class ScenarioOne(BaseOpenEI):
    def _prepare(self):
        self.name = "One"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # production maximum at a step: 1.1 * amount_need_by_all_agent_at_ste
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityOne()]


class ScenarioTwo(BaseOpenEI):
    def _prepare(self):
        self.name = "Two"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        # Generate OpenEI data and agent
        self._default()

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityTwo()]


class ScenarioThree(BaseOpenEI):
    def _prepare(self):
        self.name = "Three"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityThree()]


class ScenarioFour(BaseOpenEI):
    def _prepare(self):
        self.name = "Four"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [MultiObjectiveSum()]


class ScenarioFive(BaseOpenEI):
    def _prepare(self):
        self.name = "Five"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardOne()]


class ScenarioSix(BaseOpenEI):
    def _prepare(self):
        self.name = "Six"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardTwo()]


class ScenarioSeven(BaseOpenEI):
    def _prepare(self):
        self.name = "Seven"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = MultiObjectiveProduct
        self.rewards = [EquityRewardTwo(), Comfort(), OverConsumption()]


class ScenarioEight(BaseOpenEI):
    def _prepare(self):
        self.name = "Eight"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [Comfort()]


class ScenarioNine(BaseOpenEI):
    def _prepare(self):
        self.name = "Nine"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityOne()]


class ScenarioTen(BaseOpenEI):
    def _prepare(self):
        self.name = "Ten"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityTwo()]


class ScenarioEleven(BaseOpenEI):
    def _prepare(self):
        self.name = "Eleven"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityThree()]


class ScenarioTwelve(BaseOpenEI):
    def _prepare(self):
        self.name = "Twelve"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [MultiObjectiveSum()]


class ScenarioThirteen(BaseOpenEI):
    def _prepare(self):
        self.name = "Thirteen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardOne()]


class ScenarioFourteen(BaseOpenEI):
    def _prepare(self):
        self.name = "Fourteen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardTwo()]


class ScenarioFifteen(BaseOpenEI):
    def _prepare(self):
        self.name = "Fifteen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = MultiObjectiveProduct
        self.rewards = [EquityRewardTwo(), Comfort(), OverConsumption()]


class ScenarioSixteen(BaseOpenEI):
    def _prepare(self):
        self.name = "Sixteen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [Comfort()]


class ScenarioSeventeen(BaseOpenEI):
    def _prepare(self):
        self.name = "Seventeen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityOnePerAgent()]


class ScenarioEighteen(BaseOpenEI):
    def _prepare(self):
        self.name = "Eighteen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityTwoPerAgent()]


class ScenarioNineteen(BaseOpenEI):
    def _prepare(self):
        self.name = "Nineteen"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityThreePerAgent()]


class ScenarioTwenty(BaseOpenEI):
    def _prepare(self):
        self.name = "Twenty"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [MultiObjectiveSumPerAgent()]


class ScenarioTwentyOne(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyOne"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardPerAgent()]


class ScenarioTwentyTwo(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyTwo"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1.1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = MultiObjectiveProduct
        self.rewards = [EquityRewardPerAgent(), Comfort(), OverConsumptionPerAgent()]


class ScenarioTwentyThree(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyThree"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityOnePerAgent()]


class ScenarioTwentyFour(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyFour"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityTwoPerAgent()]


class ScenarioTwentyFive(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyFive"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityThreePerAgent()]


class ScenarioTwentySix(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentySix"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [MultiObjectiveSumPerAgent()]


class ScenarioTwentySeven(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentySeven"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardPerAgent()]


class ScenarioTwentyEight(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyEight"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = MultiObjectiveProduct
        self.rewards = [EquityRewardPerAgent(), Comfort(), OverConsumptionPerAgent()]


#  TODO Add the best two Scenario with more agent
class ScenarioTwentyNine(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyNine"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [OverConsumption()]


class ScenarioThirty(BaseOpenEI):
    def _prepare(self):
        self.name = "Thirty"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [OverConsumptionPerAgent()]


class ScenarioThirtyOne(BaseOpenEI):
    def _prepare(self):
        self.name = "Three"
        self.second_name = "ThirtyOne"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityThree()]


class ScenarioThirtyTwo(BaseOpenEI):
    def _prepare(self):
        self.name = "Four"
        self.second_name = "ThirtyTwo"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [MultiObjectiveSum()]


class ScenarioThirtyThree(BaseOpenEI):
    def _prepare(self):
        self.name = "Six"
        self.second_name = "ThirtyThree"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardTwo()]


class ScenarioThirtyFour(BaseOpenEI):
    def _prepare(self):
        self.name = "Seven"
        self.second_name = "ThirtyFour"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = MultiObjectiveProduct
        self.rewards = [EquityRewardTwo(), Comfort(), OverConsumption()]


class ScenarioThirtyFive(BaseOpenEI):
    def _prepare(self):
        self.name = "Nineteen"
        self.second_name = "ThirtyFive"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [AdaptabilityThreePerAgent()]


class ScenarioThirtySix(BaseOpenEI):
    def _prepare(self):
        self.name = "Twenty"
        self.second_name = "ThirtySix"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [MultiObjectiveSumPerAgent()]


class ScenarioThirtySeven(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyOne"
        self.second_name = "ThirtySeven"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [EquityRewardPerAgent()]


class ScenarioThirtyEight(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyTwo"
        self.second_name = "ThirtyEight"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = MultiObjectiveProduct
        self.rewards = [EquityRewardPerAgent(), Comfort(), OverConsumptionPerAgent()]


class ScenarioThirtyNine(BaseOpenEI):
    def _prepare(self):
        self.name = "TwentyNine"
        self.second_name = "ThirtyNine"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [OverConsumption()]


class ScenarioFourty(BaseOpenEI):
    def _prepare(self):
        self.name = "Thirty"
        self.second_name = "Fourty"
        self.number = {
            "office": 2,
            "school": 1,
            "residential": 10
        }
        # Generate OpenEI data and agent
        self._default()
        self.energy_generator = RandomEnergyGenerator(upper_proportion=1,
                                                      lower_cumulated=self.lower_cumulated,
                                                      upper_cumulated=self.upper_cumulated)

        self.observation_manager = ObservationManager(LocalObservation, GlobalObservation)
        self.aggregate_function = BasicAggregateFunction
        self.rewards = [OverConsumptionPerAgent()]

import os
from typing import List


from agents.agent import Agent
from environment import SmartGrid
from scenarios.scenario import Scenario


class MetricsWatcher:
    agents: List[Agent]
    smartgrid: SmartGrid

    def __init__(self, scenario: Scenario, smartgrid: SmartGrid):
        self.agents = scenario.agents
        self.smartgrid = smartgrid

    def collect(self):
        to_return = dict()

        n_agent = len(self.agents)

        to_return["available_energy"] = (self.smartgrid.available_energy, {"n_agent": n_agent})
        to_return["equity"] = (float(self.smartgrid.world.get_observation_global().equity), {"n_agent": n_agent})
        to_return["energy_loss"] = (float(self.smartgrid.world.get_observation_global().energy_loss),
                                    {"n_agent": n_agent})
        to_return["autonomy"] = (float(self.smartgrid.world.get_observation_global().autonomy), {"n_agent": n_agent})
        to_return["exclusion"] = (float(self.smartgrid.world.get_observation_global().exclusion), {"n_agent": n_agent})
        to_return["well_being"] = (float(self.smartgrid.world.get_observation_global().well_being),
                                   {"n_agent": n_agent})
        to_return["over_consumption"] = (float(self.smartgrid.world.get_observation_global().over_consumption),
                                         {"n_agent": n_agent})

        for a in self.agents:
            # creating the context
            context = {"name": a.name, "profile": a.profile.name, "n_agent": n_agent,"confort_fn": a.profile.comfort_fn.__name__}
            to_return[f"agent_state_comfort_{a.name}"] = (float(a.state.comfort), context)
            to_return[f"agent_state_payoff_{a.name}"] = (float(a.state.payoff), context)
            to_return[f"agent_state_storage_{a.name}"] = (float(a.state.storage), context)
            to_return[f"agent_grid_consumption_{a.name}"] = (float(a.enacted_action.grid_consumption), context)
            to_return[f"agent_give_energy_{a.name}"] = (float(a.enacted_action.give_energy), context)
            to_return[f"agent_storage_consumption_{a.name}"] = (float(a.enacted_action.storage_consumption), context)
            to_return[f"agent_store_energy_{a.name}"] = (float(a.enacted_action.store_energy), context)
            to_return[f"agent_buy_energy_{a.name}"] = (float(a.enacted_action.buy_energy), context)
            to_return[f"agent_sell_energy_{a.name}"] = (float(a.enacted_action.sell_energy), context)

        return to_return


class RewardWatcher:
    agents: List[Agent]

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def collect(self, rewards: List[float]):
        to_return = dict()

        for i in range(len(self.agents)):
            to_return[f"{self.agents[i].name}_reward"] = rewards[i]

        return to_return


class Collector:
    metrics_watcher: MetricsWatcher
    reward_watcher: RewardWatcher

    def __init__(self,
                 scenario: Scenario,
                 smartgrid: SmartGrid,
                 hyper_parameters_name: str,
                 model_name: str):
        self.metrics_watcher = MetricsWatcher(scenario, smartgrid)
        self.reward_watcher = RewardWatcher(smartgrid.agents)

        # hyperparameters are general
        name = scenario.name
        if scenario.second_name is not None:
            name = scenario.second_name

        self.hyper_parameters = {
                "scenario_name": name,
                "model_name": model_name,
                "hyper_parameters_name": hyper_parameters_name,
                "reward_name": str(smartgrid.world.reward_calculator),
                "energy_generator": str(scenario.energy_generator),
                "aggregate_name": scenario.aggregate_function_name,
                "data_used": str(scenario.data_conversion)
            }

    def get_path(self):
        basic_path = "./saved/"

        # create the folder if not exist
        if not os.path.exists(basic_path):
            os.mkdir(basic_path)

        basic_path += "_" + self.hyper_parameters["scenario_name"]
        basic_path += "_" + self.hyper_parameters["model_name"]
        basic_path += "_" + self.hyper_parameters["hyper_parameters_name"]

        return basic_path

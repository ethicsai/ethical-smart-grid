from typing import Type

from torch.types import Device
from tqdm import trange

from agents.agent import Action
from algorithms.model import Model
from watchers import Collector


class Runner:

    def __init__(self, hyper_parameters: dict, model: Type['Model'], device: 'Device', scenario: 'Scenario', mode):
        # principal construction
        self.mode = mode
        self.scenario = scenario
        self.smartgrid = self.scenario.make()

        # get some variable from construct object
        self.agent_num = self.smartgrid.n_agent
        self.shape = self.smartgrid.observation_shape
        self.action_dim = len(Action._fields)

        # Initialize Collector
        self.collector = Collector(scenario=self.scenario,
                                   smartgrid=self.smartgrid,
                                   hyper_parameters_name=hyper_parameters["name"],
                                   model_name=model.__name__)

        # save device used for calculation
        self.device = device

        # Initialize model
        self.model = model(self.agent_num,
                           self.smartgrid,
                           hyper_parameters,
                           self.device)

    def start(self, saving=False):
        # prepare data
        obs = self.smartgrid.reset()

        for step in trange(self.scenario.max_step):
            # collect
            self.collector.collect_metrics(step)

            # forward into the model to get all Agents Action
            actions = self.model.forward(observations_per_agent=obs)

            # pass the action to the simulator
            # and return next information
            next_obs, rewards, dones, infos = self.smartgrid.step(actions)

            self.collector.collect(infos, rewards)

            if self.mode != "evaluation":
                # reminder loop
                for agent_id in range(self.agent_num):
                    self.model.reminder(step=step, agent_id=agent_id, action=actions[agent_id],
                                        observation=obs["local"][agent_id], global_observation=obs["global"],
                                        reward=rewards[agent_id], done=dones[agent_id])

            obs = next_obs

            if self.mode != "evaluation":
                logs = self.model.backward(obs, rewards)
                self.collector.collect_logs(step, logs)

        self.smartgrid.close()
        self.collector.finalize()

        if saving:
            self.model.save(self.collector.get_path())

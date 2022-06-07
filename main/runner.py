from typing import Type

from aim import Run
from torch.types import Device

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
        # Initialize Aim Runnner
        self.aim_runner = Run()

        # Put hparams into Aim Runner: Be Careful some are reference to object inside the Simulator
        self.aim_runner['hparams'] = self.collector.hyper_parameters

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
        episode_reward = 0

        for step in range(self.scenario.max_step):
            # collect metrics for Aim
            metrics = self.collector.metrics_watcher.collect()
            for metric in metrics:
                self.aim_runner.track(value=metrics[metric][0], name=metric, step=step, context=metrics[metric][1])

            # forward into the model to get all Agents Action
            actions = self.model.forward(observations_per_agent=obs)

            # pass the action to the simulator
            # and return next information
            next_obs, rewards, dones, infos = self.smartgrid.step(actions)
            # todo put that in collector
            for agent_id in infos['rewards']:
                for reward_name in infos['rewards'][agent_id]:
                    self.aim_runner.track(name=f"{reward_name}_{agent_id}",
                                          value=infos['rewards'][agent_id][reward_name])

            # collect rewards for Aim
            aim_rewards = self.collector.reward_watcher.collect(rewards)
            for reward in aim_rewards:
                self.aim_runner.track(aim_rewards[reward], name=reward)

            # add reward
            episode_reward += sum(rewards)
            self.aim_runner.track(name="Aggregate Reward", value=sum(rewards))

            if self.mode != "evaluation":
                # reminder loop
                for agent_id in range(self.agent_num):
                    self.model.reminder(step=step, agent_id=agent_id, action=actions[agent_id],
                                        observation=obs["local"][agent_id], global_observation=obs["global"],
                                        reward=rewards[agent_id], done=dones[agent_id])

            obs = next_obs

            if self.mode != "evaluation":
                logs = self.model.backward(obs, rewards)
                for log in logs:
                    self.aim_runner.track(value=logs[log], name=log, step=step)

        self.smartgrid.close()

        if saving:
            self.model.save(self.collector.get_path())

        self.aim_runner.finalize()

"""
This module defines a Q-SOM helper that is used as an entrypoint to simplify
the instantiation of Q-SOM Agents from a Gym Environment.

It handles creating the correct structures, giving the correct parameters, ...
"""
from algorithms.qsom.qsom_agent import QsomAgent
from algorithms.qsom.som import SOM
from algorithms.util.action_perturbator import EpsilonActionPerturbator
from algorithms.util.action_selector import BoltzmannActionSelector
from smartgrid.environment import SmartGrid


class QSOM(object):

    def __init__(self, env: SmartGrid,
                 q_learning_rate: float = 0.7,
                 q_discount_factor: float = 0.9,
                 update_all: bool = True,
                 use_neighborhood: bool = True,
                 sigma_state: float = 1.0,
                 lr_state: float = 0.58,
                 sigma_action: float = 1.0,
                 lr_action: float = 0.7,
                 initial_tau: float = 0.5,
                 tau_decay: bool = False,
                 tau_decay_coeff: float = 1.0,
                 noise: float = 0.08):
        self.n_agents = env.n_agent
        self.qsom_agents = []

        action_selector = BoltzmannActionSelector(initial_tau, tau_decay, tau_decay_coeff)
        action_perturbator = EpsilonActionPerturbator(noise)

        for num_agent in range(self.n_agents):
            obs_space = env.observation_space[num_agent]
            assert len(obs_space.shape) == 1, 'Observation space must be 1D'
            action_space = env.action_space[num_agent]
            assert len(action_space.shape) == 1, 'Action space must be 1D'

            state_som = SOM(12, 12,
                            obs_space.shape[0],
                            sigma=sigma_state,
                            learning_rate=lr_state)
            action_som = SOM(3, 3,
                             action_space.shape[0],
                             sigma=sigma_action,
                             learning_rate=lr_action)

            qsom_agent = QsomAgent(obs_space,
                                   action_space,
                                   state_som,
                                   action_som,
                                   action_selector,
                                   action_perturbator,
                                   q_learning_rate=q_learning_rate,
                                   q_discount_factor=q_discount_factor,
                                   update_all=update_all,
                                   use_neighborhood=use_neighborhood)

            self.qsom_agents.append(qsom_agent)

    def forward(self, observations_per_agent):
        """Choose an action for each agent, based on their observations."""
        assert len(observations_per_agent) == self.n_agents
        actions = [
            self.qsom_agents[i].forward(observations_per_agent[i])
            for i in range(self.n_agents)
        ]
        return actions

    def backward(self, new_observations_per_agent, reward_per_agent):
        """Make each agent learn, based on their rewards and observations."""
        assert len(reward_per_agent) == self.n_agents
        assert len(new_observations_per_agent) == self.n_agents
        for i, agent in enumerate(self.qsom_agents):
            agent.backward(new_observations_per_agent[i],
                           reward_per_agent[i])

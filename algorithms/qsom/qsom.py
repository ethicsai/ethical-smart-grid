"""
This module defines a Q-SOM helper that is used as an entrypoint to simplify
the instantiation of Q-SOM Agents from a Gym Environment.

It handles creating the correct structures, giving the correct parameters, ...
"""
import numpy as np

from algorithms.model import Model
from algorithms.qsom.qsom_agent import QsomAgent
from algorithms.qsom.som import SOM
from algorithms.util.action_perturbator import EpsilonActionPerturbator
from algorithms.util.action_selector import BoltzmannActionSelector
from smartgrid.environment import SmartGrid


class QSOM(Model):
    """
    The Q-SOM learning algorithm: based on Q-Learning + Self-Organizing Maps.

    Two SOMs are used, a State-SOM that learns to handle the observation (state)
    space, i.e., to map continuous observations to discrete space identifiers;
    and an Action-SOM that learns to handle the action space, i.e., to map
    discrete actions identifiers to continuous action parameters.

    A Q-Table learns the interests of (discrete) actions in (discrete) states.

    List of hyperparameters that this model expects:

    initial_tau
        Initial value of the Boltzmann temperature, which controls the
        exploration-exploitation trade-off.

    tau_decay
        Whether to decrease (decay) the Boltzmann temperature over the time
        steps, so as to encourage exploitation rather than exploration in
        later time steps. See also the ``tau_decay_coeff`` below.

    tau_decay_coeff
        Coefficient of reduction of the Boltzmann temperature, each step,
        if the decay is enabled. Applied multiplicatively to the current
        tau each time step, i.e., ``tau = tau * tau_decay_coeff``.

    noise
        The noise parameter that controls the random distribution when
        perturbing actions. The higher the noise, the more the action will
        be perturbed (i.e., far from its original, unperturbed version).

    sigma_state
        Size of the neighborhood for the State-SOM.

    sigma_action
        Size of the neighborhood for the Action-SOM.

    lr_state
        Learning rate for the State-SOM.

    lr_action
        Learning rate for the Action-SOM.

    q_learning_rate
        Learning rate for the Q-Table.

    q_discount_factor
        The gamma value controls the horizon of expected rewards: the higher
        it is, the more the agent will take into account the future states,
        and rewards that can be expected from these future states, when
        determining its policy. If set to 0, the agent will simply maximize
        the current expected reward (greedy policy).

    update_all
        Whether to update all Q-Values (Smith's optimization) at each step.
        This speeds up the learning of interests.

    use_neighborhood
        Whether to use the State- and Action-SOMs neighborhoods when updating
        the Q-Values.
    """

    default_hyperparameters = {
        "q_learning_rate": 0.7,
        "q_discount_factor": 0.9,
        "update_all": True,
        "use_neighborhood": True,
        "sigma_state": 1.0,
        "lr_state": 0.8,
        "sigma_action": 1.0,
        "lr_action": 0.7,
        "initial_tau": 0.5,
        "tau_decay": False,
        "tau_decay_coeff": 1.0,
        "noise": 0.08
    }

    def __init__(self, env: SmartGrid, hyper_parameters: dict = None):
        if hyper_parameters is None:
            hyper_parameters = QSOM.default_hyperparameters
        super().__init__(env, hyper_parameters)
        self.qsom_agents = []

        action_selector = BoltzmannActionSelector(self.hyper_parameters["initial_tau"],
                                                  self.hyper_parameters["tau_decay"],
                                                  self.hyper_parameters["tau_decay_coeff"])
        action_perturbator = EpsilonActionPerturbator(self.hyper_parameters["noise"])

        for num_agent in range(env.n_agent):
            obs_space = env.observation_space[num_agent]
            assert len(obs_space.shape) == 1, 'Observation space must be 1D'
            action_space = env.action_space[num_agent]
            assert len(action_space.shape) == 1, 'Action space must be 1D'

            state_som = SOM(12, 12,
                            obs_space.shape[0],
                            sigma=self.hyper_parameters["sigma_state"],
                            learning_rate=self.hyper_parameters["lr_state"])
            action_som = SOM(3, 3,
                             action_space.shape[0],
                             sigma=self.hyper_parameters["sigma_action"],
                             learning_rate=self.hyper_parameters["lr_action"])

            qsom_agent = QsomAgent(obs_space,
                                   action_space,
                                   state_som,
                                   action_som,
                                   action_selector,
                                   action_perturbator,
                                   q_learning_rate=self.hyper_parameters["q_learning_rate"],
                                   q_discount_factor=self.hyper_parameters["q_discount_factor"],
                                   update_all=self.hyper_parameters["update_all"],
                                   use_neighborhood=self.hyper_parameters["use_neighborhood"])

            self.qsom_agents.append(qsom_agent)

    def forward(self, observations_per_agent):
        """Choose an action for each agent, based on their observations."""
        observations_per_agent = [
            np.concatenate((
                observations_per_agent['local'][i],
                observations_per_agent['global'],
            ))
            for i in range(self.env.n_agent)
        ]
        assert len(observations_per_agent) == len(self.qsom_agents)
        actions = [
            self.qsom_agents[i].forward(observations_per_agent[i])
            for i in range(len(self.qsom_agents))
        ]
        return actions

    def backward(self, new_observations_per_agent, reward_per_agent):
        """Make each agent learn, based on their rewards and observations."""
        new_observations_per_agent = [
            np.concatenate((
                new_observations_per_agent['local'][i],
                new_observations_per_agent['global'],
            ))
            for i in range(self.env.n_agent)
        ]
        assert len(reward_per_agent) == len(self.qsom_agents)
        assert len(new_observations_per_agent) == len(self.qsom_agents)
        for i, agent in enumerate(self.qsom_agents):
            agent.backward(new_observations_per_agent[i],
                           reward_per_agent[i])

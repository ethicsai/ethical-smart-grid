"""
This module implements a Q-SOM Agent, with the decision and learning
algorithms that make the agent act based on the received observations
from the environment.

The Q-SOM Agent uses 2 SOMs to represent the continuous and multidimensional
States and Actions.
"""

import numpy as np
from gymnasium.spaces import Box

from algorithms.qsom.som import SOM
from algorithms.util.action_perturbator import ActionPerturbator
from algorithms.util.action_selector import ActionSelector
from smartgrid.util import interpolate


class QsomAgent(object):

    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 state_som: SOM,
                 action_som: SOM,
                 action_selector: ActionSelector,
                 action_perturbator: ActionPerturbator,
                 q_learning_rate=0.7,
                 q_discount_factor=0.9,
                 update_all=True,
                 use_neighborhood=True
                 ):
        """
        Initialize an Agent using the Q-SOM learning and decision algorithm.
        """

        # The State Map (observations -> discrete state)
        self.observation_space = observation_space
        self.state_som = state_som

        # The Action Map (discrete action ID -> vector of action parameters)
        self.action_space = action_space
        self.action_som = action_som

        # Q-Table: Expected interest (i.e. Q-Value) of an action in a state
        self.qtable = np.zeros((self.state_som.nb_units,
                                self.action_som.nb_units),
                               dtype=np.longdouble)
        # Memorize the number of "hits" on each cell of the Q-Table
        self.hits = np.zeros(self.qtable.shape, dtype=int)

        self.experiences = []

        self.action_selector = action_selector
        self.action_perturbator = action_perturbator

        # Q-Learning parameters
        self.alpha = q_learning_rate  # α  = Q-Learning Rate
        self.gamma = q_discount_factor  # γ  = Q-Learning Discount Factor
        self.update_all = update_all
        self.use_neighborhood = use_neighborhood

        # We memorize data (observations, chosen action) at t, in order to
        # reuse it at t+1
        self.last_input = None
        self.last_input_idx = None
        self.last_action = None
        self.last_action_idx = None
        self.step = 0

    def forward(self, observations):
        # This method follows Smith's algorithm. The steps are identified for
        # easier readability

        # Interpolate observations from their space to [0,1]^n (easier for SOM)
        observations = np.asarray(observations)
        observations = self._interpolate_observations(observations)

        # 1. Identify winner neuron in the input map
        input_idx = self.state_som.compute_winner_node(observations)

        # 2-3. Identify proposed action in the action map
        action_idx = self.action_selector(self.qtable[input_idx], self.step)
        self.hits[input_idx][action_idx] += 1
        action_unit = self.action_som.get_unit(action_idx)

        # 4. Perturb the action (trial-and-error exploration of the Action Map)
        action_unit = self.action_perturbator(action_unit)

        # Memorize the values, so we can learn when we get the reward
        self.last_input = observations
        self.last_input_idx = input_idx
        self.last_action = action_unit
        self.last_action_idx = action_idx

        # 5. Take the action in the Environment (first interpolate)
        action_unit = self._interpolate_action(action_unit)

        return action_unit

    def backward(self, new_perception: np.ndarray, reward: float):
        # 7. Check if perturbed action is an improvement over the existing action
        # <=> (r + γ max_i Q[sj',ai]) > Q[sj,ak]
        new_state = self.state_som.compute_winner_node(new_perception)
        max_reward = np.max(self.qtable[new_state])
        qvalue = self.qtable[self.last_input_idx][self.last_action_idx]

        # Log the experience (state, action, new state, reward)
        self.experiences.append({'perceptions': self.last_input,
                                 'state': self.last_input_idx,
                                 'action_id': self.last_action_idx,
                                 'action': self.last_action,
                                 'reward': reward,
                                 'new_perceptions': new_perception,
                                 'new_state': new_state})

        if (reward + self.gamma * max_reward) > qvalue:
            # Update the Action SOM
            # The pattern to learn (data) is the perturbed action
            # The winning node is the proposed action (non-perturbed) index
            self.action_som.update(self.last_action, self.last_action_idx)

        # 8. Update Q-Values
        # Q[sm,an] += α*φS(j,m,NS)*φA(k,n,NA)*(r + γ max_i Q[sj',ai] - Q[sm,an])
        self._update_qvalues(reward, max_reward)

        # 9. Update the Input SOM
        self.state_som.update(self.last_input, self.last_input_idx)

        self.step += 1

    def _update_qvalues(self, reward: float, max_reward: float):
        # Compute the neighborhood of Input- and Action-SOM
        # (i.e. the φS and φA in the update formula)
        if self.use_neighborhood:
            input_neigh = self.state_som.neighborhood(self.last_input_idx)
            action_neigh = self.action_som.neighborhood(self.last_action_idx)
        # Determine which Q-Values to update
        if self.update_all:
            # All Q-Values
            states = range(self.state_som.nb_units)
            actions = range(self.action_som.nb_units)
        else:
            # Only the (state,action) pair that was used this step
            states = [self.last_input_idx]
            actions = [self.last_action_idx]
        # Update the Q-Values
        for s in states:
            if self.use_neighborhood:
                psi_s = input_neigh[self.state_som.coords_map[s]]  # <=> φS(j,s,NS)
            for a in actions:
                if self.use_neighborhood:
                    psi_a = action_neigh[self.action_som.coords_map[a]]  # <=> φA(k,a,NA)
                # Δ = α*(r + γ*max_i Q[sj',ai] - Q[sm,an])
                delta = self.alpha * (reward + self.gamma * max_reward - self.qtable[s][a])
                if self.use_neighborhood:
                    # Δ = α*φS(j,s,NS)*φA(k,a,NA)*(r + γ*max_i Q[sj',ai] - Q[sm,an])
                    delta *= (psi_s * psi_a)
                self.qtable[s][a] += delta

    def _interpolate_observations(self, observations: np.ndarray):
        """
        Interpolate observations from their space to the [0,1]^n space.

        It is easier for SOMs to handle values constrained to the [0,1]^n
        space, thus we need to interpolate them from any (bounded) space.
        For example, if the original space is [0,100]x[0,200], the value
        [40, 150] interpolated in [0,1]^2 (or [0,1]x[0,1]) is: [0.4, 0.75].
        The original observation space is known to this agent as the
        `self.observation_space` attribute.
        """
        assert len(self.observation_space.shape) == 1, 'Observation space must be 1D'
        old_bounds = list(zip(self.observation_space.low, self.observation_space.high))
        new_bounds = [(0.0, 1.0)] * self.observation_space.shape[0]
        return interpolate(observations, old_bounds, new_bounds)

    def _interpolate_action(self, action: np.ndarray):
        """
        Interpolate action from the [0,1]^n space to their space.

        Similarly to the observations, it is easier for SOMs to handle
        actions constrained to the [0,1]^n space. However, since actions
        are produced by SOMs, we interpolate in the other direction.
        """
        assert len(self.action_space.shape) == 1, 'Action space must be 1D'
        old_bounds = [(0.0, 1.0)] * self.action_space.shape[0]
        new_bounds = list(zip(self.action_space.low, self.action_space.high))
        return interpolate(action, old_bounds, new_bounds)

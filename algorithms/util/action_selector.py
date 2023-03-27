"""
This module defines several classes to select actions (ActionSelectors).

An ActionSelector takes a list of interests (e.g., Q-Values) and the time step,
to return a single identifier, which is considered the selected action.
They target the exploration-exploitation dilemma.

We consider 2 selectors:

- the Epsilon-Greedy selector selects the maximum interest action with a
  `(1-ε)` probability, e.g., 95%. Otherwise, it selects a random action.
- the Boltzmann selector applies a Boltzmann distribution over the interests.
  Interests that are closer have a similar probability, and higher interests
  yield higher probabilities. The distribution is controlled by a Boltzmann
  temperature, such that low interests can still yield significant probabilities.
"""

from random import random, randrange, choices

import numpy as np


class ActionSelector(object):

    def choose(self, interests, step) -> int:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.choose(*args, **kwargs)


class EpsilonGreedyActionSelector(ActionSelector):
    """Implements the ε-greedy policy."""

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def choose(self, interests, step):
        if random() < self.epsilon:
            # Exploration: pick a random unit
            action_idx = randrange(0, len(interests))
        else:
            # Exploitation: pick the unit with the maximal Q-Value
            action_idx = np.argmax(interests)
        return action_idx


class BoltzmannActionSelector(ActionSelector):
    """Implements the Boltzmann policy."""

    def __init__(self,
                 initial_tau: float,
                 tau_decay: bool,
                 tau_decay_coeff: float):
        self.initial_tau = initial_tau
        self.tau_decay = tau_decay
        self.tau_decay_coeff = tau_decay_coeff

    def choose(self, values, step):
        # Boltzmann decision process
        # First, compute tau (τ)
        if self.tau_decay:
            tau = self.initial_tau * (self.tau_decay_coeff ** step)
            tau = max(tau, 0.01)
        else:
            tau = self.initial_tau
        # Then, compute the weight for each value (exp(Q[s,a]) / τ)
        indices = np.arange(len(values))
        weights = [np.exp(values[i] / tau) for i in indices]
        return choices(indices, weights=weights, k=1)[0]

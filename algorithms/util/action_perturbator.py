"""
This module defines several classes to perturb (explore) actions.

An ActionPerturbator takes a continuous action, i.e., a vector of parameters,
and returns a vector with same shape but different values.
The difference between the original vector and the resulting one is called
the perturbation, is often implemented as a random noise, and allows exploring
the action space.

We have implemented 4 perturbators:

- Epsilon applies a uniform random noise to all dimensions.
- Gaussian applies a gaussian random noise to all dimensions.
- MultiDim has a probability to apply a uniform noise to each dimension.
- Identity performs no perturbation.
"""

import numpy as np


class ActionPerturbator(object):

    def perturb(self, action, clip=True):
        """
        Add a noise to the proposed action.

        :param action: A 1D vector of values representing the action.
        :type action: np.ndarray

        :param clip: Controls whether to clip the noised values in [0,1].

        :return: A vector of noised values, with the same shape as `action`.
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class EpsilonActionPerturbator(ActionPerturbator):
    """Implements a simple ε noise."""

    def __init__(self, noise: float):
        self.noise = noise

    def perturb(self, action, clip=True):
        # Epsilon-based perturbation
        noise = np.random.uniform(-self.noise, self.noise, len(action))
        action += noise
        if clip:
            action = np.clip(action, 0.0, 1.0)
        return action


class GaussianActionPerturbator(ActionPerturbator):
    """Implements a Gaussian (σ) noise."""

    def __init__(self, noise: float):
        self.noise = noise

    def perturb(self, action, clip=True):
        # Gaussian-based perturbation
        noise = np.random.normal(0, self.noise, len(action))
        action = action + noise
        if clip:
            action = np.clip(action, 0.0, 1.0)
        return action


class MultiDimActionPerturbator(ActionPerturbator):
    """Custom algorithm, with a probability to noise each dimension."""

    def __init__(self, noise: float, probability: float):
        self.noise = noise
        self.proba = probability

    def perturb(self, action, clip=True):
        # Draw a die for each dimension to apply (or not) a random noise
        for k in range(len(action)):
            if np.random.random() < self.proba:
                # Apply noise to this dimension
                noise = np.random.uniform(-self.noise, self.noise)
                action[k] = action[k] + noise
                if clip:
                    action[k] = np.clip(action[k], 0.0, 1.0)
        return action


class IdentityAction(ActionPerturbator):
    """Returns the same action (no perturbation)."""

    def perturb(self, action, clip=True):
        if clip:
            action = np.clip(action, 0.0, 1.0)
        return action

"""
Helper classes that can be used by all learning algorithms.

In particular:

- ``action_perturbator`` can be used to randomly noise (*perturb*) an action.
- ``action_selector`` can be used to select an action, based on its interest,
  and a given distribution. These classes target the exploration-exploitation
  dilemma: finding a trade-off between selecting the action with the best
  interests, and trying other actions to discover their true interests.
"""


from .action_perturbator import (ActionPerturbator,
                                 EpsilonActionPerturbator,
                                 GaussianActionPerturbator,
                                 MultiDimActionPerturbator,
                                 IdentityAction)

from .action_selector import (ActionSelector,
                              EpsilonGreedyActionSelector,
                              BoltzmannActionSelector)

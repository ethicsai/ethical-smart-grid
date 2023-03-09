"""
Numeric reward functions are purely based on mathematical expressions.

They come in two variations:

- Difference Rewards propose to compute both the current state of the
  environment, and a hypothetical environment in which the agent would have
  not acted. This allows to finely determine the agent's contribution: if
  the current environment is better than the hypothetical one, we can say
  the agent has improved the environment by its action. Otherwise, if the
  current environment is worse than the hypothetical one, the agent's action
  has degraded the environment.

- Local or "per agent" rewards focus on a single agent. They do not compare
  to the society of agents. They can be useful when the learning algorithm
  itself already tries to learn "difference" rewards (e.g., COMA).
"""

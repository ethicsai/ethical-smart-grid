"""
Wrapper that defines Env-level regimentation of norms.

Norms cannot be violated; if an action violates a norm, this action is
prevented from happening by the Env (SmartGrid), and a default action
(of doing nothing, i.e., ``[0, 0, 0, 0, 0, 0]``) happens instead.
"""

from typing import Iterable, Callable, Any

import gymnasium as gym

from smartgrid.environment import SmartGrid
from smartgrid.agents import Action


class NormsWrapper(gym.Wrapper):

    default_action = Action(0, 0, 0, 0, 0, 0)

    def __init__(self,
                 env: SmartGrid,
                 # TODO: Maybe norms should take a situation (obs) and an action as params!
                 # TODO: Maybe norms should be a class, so that we can provide a name,
                 #  a description (?), a test function, a string representation, ...
                 norms: Iterable[Callable[[Any], bool]],
                 remove_violation_rewards: bool):
        super().__init__(env)
        self.norms = norms
        self.remove_violation_rewards = remove_violation_rewards

    def step(self, actions):
        """
        Prevents violating actions from happening and executes a step.

        :param actions: The actions that decision-making algorithms intend to
            perform. They will be filtered internally, see
            :py:meth:`~smartgrid.wrappers.norms.NormsWrapper._filter_actions`.

        :return: A tuple with same shape as the standard Gymnasium API:
            ``(observations, rewards, terminated, truncated, info)``.
            However, 2 modifications are performed:
            - ``rewards``: if :py:attr:`~.remove_violation_rewards` is ``True``,
              the ``rewards`` object is modified to replace rewards of agents
              whose action was violating a norm with ``None``. In other words,
              if ``actions[3]`` violates a norm (``self._check_action(actions[3])
              == False``), then ``rewards[3] == None``.
            - ``info``: a new field ``violations`` is added, which is the list
              representing whether each action violated the norms.
        """
        violates, new_actions = self._filter_actions(actions)
        obs, reward, terminated, truncated, info = self.env.step(new_actions)
        info['violations'] = violates
        if self.remove_violation_rewards:
            # Replace all rewards from violating agents with `None`
            # TODO: This would have a better type if we could switch to PettingZoo...
            reward = [
                reward[i] if not violates[i] else None
                for i in range(len(reward))
            ]
        return obs, reward, terminated, truncated, info

    def _filter_actions(self, actions):
        """
        Determine which actions violate norms and replace them by default actions.

        :param actions: An Iterable of actions, where each action should be
            a list of parameters.

        :return: The two following values:
            - ``violates``: a list indicating, for each action, if it violates
              at least of one the norms.
            - ``new_actions``: the list of filtered actions, where each element
              is either the action that was passed as input, if it does not
              violate norms, or a default action, if the action violated a
              norm. Filtered actions correspond to actions with same index;
              e.g., if ``actions[3]`` violated a norm, ``new_actions[3]`` will
              be a default action. On the contrary, if ``actions[1]`` did not
              violate any norm, then ``new_actions[1] == actions[1]``.
        """
        violates = []
        new_actions = []
        for action in actions:
            action_is_violating = not self._check_action(action)
            violates.append(action_is_violating)
            new_actions.append(
                NormsWrapper.default_action if action_is_violating else action
            )
        return violates, new_actions

    def _check_action(self, action):
        """
        Check that a single action respects all norms.

        :param action: The action that should be checked, a list of parameter.
            (Ideally a NumPy ndarray, or at the very least a Python list of
            floats, e.g., ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]``).

        :return: ``True`` if the action is validated by *all* norms, and
            ``False`` otherwise (i.e., if at least one norm is violated).
        """
        for norm in self.norms:
            # TODO: Maybe a norm should return an object Violation which
            #  could optionally describe why/how the action violates it. Then
            #  this action would return an Optional[Violation], None meaning
            #  that there is no violation. The info object could collect all
            #  violations instances for each action, instead of simply True/False.
            if not norm(action):
                return False
        return True

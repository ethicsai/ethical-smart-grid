"""
A JudgingAgent judges the behaviour of learning agents, w.r.t a moral value.

This code is mostly based on the work of Benoît Alcaraz.
"""

import copy
from dataclasses import dataclass
from typing import Dict, Callable

from .afdm import AFDM


# We want to be able to mutate this class, thus `frozen` is set to `False`.
@dataclass(frozen=False)
class Activation:
    """
    Memorizes the activation rates of arguments (for post-run analysis).
    """

    activated: int = 0
    """
    Number of times an argument was activated, i.e., both alive in a given
    situation and part of the grounded extension.
    """

    total: int = 0
    """
    Total number of time steps the argument was present in the graph.
    """

    @property
    def proportion(self) -> float:
        """
        Proportion of time steps in which the argument was activated.
        """
        if self.total != 0:
            return self.activated / self.total
        else:
            return 0.0


# Note: no dot after `w.r.t`, otherwise Sphinx would assume it is the end of
# the line, and scratch the rest of the sentence in the summary tables...
class JudgingAgent:
    """
    A JudgingAgent judges the behaviour of learning agents, w.r.t a moral value.

    This moral value is implemented (represented) by an
    :py:class:`~smartgrid.rewards.argumentation.lib.afdm.AFDM` and its arguments
    and attacks. The judgment of a given action in a situation is performed
    based on the AFDM, by computing its *grounded extension*, i.e., arguments
    that are *alive* in a given situation and that defend each other.

    The JudgingAgent also contains a function
    :py:attr:`~smartgrid.rewards.argumentation.lib.judging_agent.JudgingAgent.j_transform`
    to transform the grounded extension into a scalar measure, i.e., the
    *reward*. This function is configurable, and often relies on comparing the
    number of *pros* arguments (that support the idea that the learning agent's
    action was moral) and *cons* arguments (that counter the idea that the
    action was moral).
    """

    moral_value: str
    """
    The name of the moral value that this Judging Agent represents.

    It should be unique, so that it can be used as key of dicts.
    """

    base_afdm: AFDM
    """
    The base AFDM, containing all arguments and attacks representing the moral value.

    *All* arguments means here that we consider arguments for all possible
    situations, e.g., "agent bought 10% of energy", "agent bought energy 20%",
    etc. Some of them will be "false" in some situations, e.g., if the agent
    bought in fact 15% of energy, the first argument is true (*alive*) but not
    the second one. The base AFDM contains all of them, and is thus refined
    by updating the arguments' :py:attr:`~.Argument.alive` attribute in specific
    situations.
    """

    j_transform: Callable[[AFDM, str], float]
    """
    The function that transforms the grounded extension into a scalar reward.

    It is responsible for ultimately producing the reward, when the
    argumentation process is done and that only arguments that are *acceptable*
    remain, in the grounded extension. From this set of arguments, the
    ``j_transform`` determines how to measure the degree to which the learning
    agent's action was moral. A simple version can be to compare the number
    of *pros* arguments to the total number of *pros* and *cons* arguments.

    The ``j_transform`` function takes as a first input the *current AFDM* that
    represents the current situation and the grounded extension, and as a
    second input the *decision* that we want to measure. In the current version,
    the decision will always be ``'moral'``, but in future versions it could be
    extended to support various decisions, ethical principles, etc.
    It then returns a float.
    """

    activation_rate: Dict[str, Activation]
    """
    The proportions of situations in which each argument was activated.

    Activation rates are indexed by the arguments' name. They contain the
    number of times the argument was activated, i.e., in the grounded
    extension (``activated``); and the total number of steps (``total``).
    In other words, the proportion of an argument named ``k`` is 
    ``activation_rate[k].activated / activation_rate[k].total``. For easier
    access, this is also available as ``activation_rate[k].proportion``.

    Activation rates are only useful for post-run analysis and are not taken
    into account during the judgment process.
    """

    last_activation: Dict[str, bool]
    """
    For each argument, whether it was activated at the last step.
    """

    def __init__(self, moral_value: str,
                 base_afdm: AFDM,
                 j_transform: Callable[[AFDM, str], float]):
        """
        Create a new JudgingAgent.

        :param moral_value: The name of the moral value associated to this
            judging agent. Should be unique.
        :param base_afdm: The argumentation framework (arguments + attacks)
            used to implement or represent the moral value.
        :param j_transform: The function used to transform the grounded
            extension into a scalar reward. See the
            :py:mod:`~smartgrid.rewards.argumentation.lib.judgment` module for
            examples of such functions.
        """
        self.moral_value = moral_value.lower()
        self.activation_rate = {}
        self.last_activation = {}
        self.base_afdm = copy.deepcopy(base_afdm)
        self.j_transform = j_transform
        # Initialize the `activation_rate` dict for each argument.
        for argument in self.base_afdm.arguments:
            self.activation_rate[argument.name] = Activation(0, 0)

    def judge(self, situation, decision: str) -> float:
        """
        Judge a situation and return the corresponding reward.

        :param situation: A situation corresponds to an action that happened in
            a state, but does not have any "structured" type (it can be Any) to
            allow for flexibility. The situation should describe both the state
            and the action, and will usually be a dict, indexed by strings. See
            :py:attr:`~.Argument.activation_function` for details.
        :param decision: The decision with respect to which the judgment should
            be done. In the current version, a single decision ``'moral'`` is
            used; in future versions, different decisions could be used to
            implement various ethical principles.

        :return: The reward associated to the given situation, as determined
            through this agent's judgment, with respect to its moral value.
        """
        # Afdm specialized to the current situation (arguments alive/killed).
        current_afdm = self._filter_afdm(situation)
        # Afdm update w.r.t. the grounded extension.
        current_afdm.update_alive_in_grounded_extension()
        # We want to memorize the number of times an argument was activated
        # in order to provide stats (post-simulation analysis).
        self._update_activation_rate(current_afdm)
        # Finally, transform the grounded extension into a scalar reward.
        reward = self.j_transform(current_afdm, decision)
        return reward

    def _filter_afdm(self, situation) -> AFDM:
        """
        Filters the base AFDM to return the current AFDM.

        The current (or *filtered*) AFDM contains only arguments considered
        to be alive in the current situation. See
        :py:attr:`~.Argument.activation_function` for details.

        :param situation: The current situation. Its type is voluntarily not
            specified, to avoid restricting users. The recommended type is
            a dictionary, mapping string (keys) to any object, thus representing
            the current situation as a list of properties.

        :return: The new AFDM that corresponds to the current situation.
        """
        current_afdm = AFDM()
        # First, copy all arguments and set their aliveness based on this
        # situation. We need to copy all arguments, even those that are not
        # alive, because some judgment functions compare the *total* number
        # of supporting/countering arguments, i.e., alive or not.
        for argument in self.base_afdm.arguments:
            new_argument = current_afdm.add_argument(argument)
            new_argument.alive = new_argument.compute(situation)
        # Now, copy the attack relationships that consider alive arguments.
        for attack in self.base_afdm.R:
            if current_afdm.is_alive(attack.attacker) and \
                    current_afdm.is_alive(attack.attacked):
                current_afdm.add_attack_relation(attack.attacker, attack.attacked)

        return current_afdm

    def _update_activation_rate(self, current_afdm: AFDM):
        """
        Update the arguments' activation rates, based on the grounded extension.

        Arguments that are in the grounded are said to be *activated*, and we
        increment both counters (*activated* and *total*). For other arguments,
        we only increment the *total* counter, such that their proportion of
        activation decreases.
        """
        self.last_activation = {}
        for argument in self.base_afdm.arguments:
            in_grounded = argument.name in current_afdm.grounded
            # We remember the activation of each argument at the last step.
            self.last_activation[argument.name] = in_grounded
            # Just in case, if any argument is new, we create an entry.
            if argument.name not in self.activation_rate.keys():
                self.activation_rate[argument.name] = Activation(0, 0)
            # We transform the bool into an int, so that we can increase the
            # counter: `int(True) == 1` so we effectively do `+= 1` when in
            # the grounded. On the contrary, `int(False) == 0`.
            self.activation_rate[argument.name].activated += int(in_grounded)
            self.activation_rate[argument.name].total += 1

    def __str__(self):
        return f'<JudgingAgent {self.moral_value}>'

    __repr__ = __str__

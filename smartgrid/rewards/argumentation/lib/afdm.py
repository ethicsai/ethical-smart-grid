"""
Defines the Argumentation-Framework for Decision-Making and Attacks.

This code is mostly based on the work of Benoît Alcaraz.
"""

import copy
import warnings
from collections import defaultdict
from typing import List, Dict, Union, Set, Iterable, Optional

from .argument import Argument
from .attack import Attack
from .exceptions import ArgumentNotFoundError


class AFDM:
    """
    An Argumentation Framework for Decision-Making.

    An AFDM contains arguments, and an attack (binary) relationship between
    arguments. It can be used to compute the grounded extension of arguments in
    a given situation (i.e., with only certain arguments activated).
    """

    A: Dict[str, Argument]
    """
    The map of known arguments, indexed by their name for faster access.

    To iterate over arguments, please see :py:attr:`~.arguments`.
    """

    attackers: Dict[str, Set[str]]
    """
    A dictionary mapping arguments to their attackers.

    In other words, ``attackers[x] = { y, z }`` mean that ``y`` and ``z`` are
    attackers of ``x``. More generally, it should be read as:
    ``attackers[attacked] = set of attackers``.

    This structure (dict of sets) is also known as a MultiMap (or MultiSet),
    and is optimal for performances, when getting all attackers of a given
    argument, instead of iterating over the whole list of attack relations.
    The :py:meth:`.get_attackers` method offers is the public interface for
    getting attackers in an easy way (and also allows to return Arguments
    instead of their names).

    .. warning:: This field should **never** be modified directly! Otherwise,
        the :py:attr:`.attacked` attribute would not be updated, and the
        attack relationship would not be correctly set. Please use the
        :py:meth:`.add_attack_relation` method instead, which ensures both
        attribute are modified.
    """

    attacked: Dict[str, Set[str]]
    """
    A dictionary mapping arguments to the arguments that they attack.

    In other words, ``attacked[x] = { y, z }`` means that ``x`` attacks ``y``
    and ``z``. More generally, it should be read as ``attacked[attacker] = set
    of attacked arguments``.

    This structure (dict of sets) is also known as a MultiMap (or MultiSet),
    and is optimal for performances, when getting all attacked arguments of a
    given argument, instead of iterating over the whole list of attack
    relations. The :py:meth:`.get_attacked` method offers is the public
    interface for getting attackers in an easy way (and also allows to return
    Arguments instead of their names).

    .. warning:: This field should **never** be modified directly! Otherwise,
        the :py:attr:`.attackers` attribute would not be updated, and the
        attack relationship would not be correctly set. Please use the
        :py:meth:`.add_attack_relation` method instead, which ensures both
        attribute are modified.
    """

    decisions: Set[str]
    """
    The set of known decisions.

    Decisions are retrieved from the arguments'
    :py:attr:`~smartgrid.rewards.argumentation.lib.argument.Argument.support`
    and :py:attr:`~smartgrid.rewards.argumentation.lib.argument.Argument.counter`
    fields. As mentioned in their docs, in our current version, decisions are
    not really used. In practice, the default ``'moral'`` decision will be the
    only one, i.e., arguments support or counter "The agent's last action was
    moral with respect to the given moral value".
    """

    grounded: List[str]
    """
    The list of arguments' names that are in the grounded extension at the
    current step.

    The grounded extension must be re-computed each step, by taking into
    account the new situation, and thus the new *aliveness* of arguments.
    """

    def __init__(self):
        """
        Create a new AFDM.
        """
        self.A = {}
        self.attackers = defaultdict(set)
        self.attacked = defaultdict(set)
        self.decisions = set()
        self.grounded = []

    @property
    def arguments(self) -> Iterable[Argument]:
        """
        All arguments known in this AFDM.

        This is a shortcut to ``self.A.values()``, allowing to access the
        arguments themselves (not the dict) easily.
        """
        return self.A.values()

    @property
    def R(self) -> Iterable[Attack]:
        """
        All attack relations known in this AFDM.

        To simplify usage and enforce types, attacks are represented by the
        Attack dataclass. In the comments, we will also note them as tuples,
        e.g., ``('a', 'b')`` means that argument ``'a'`` attacks ``'b'``.

        This should be used to iterate over all attacks, but not when accessing
        the attackers of a specific argument (or, conversely, the arguments
        attacked by a specific argument). For these operations, the optimized
        way is to use the :py:meth:`.get_attackers` and :py:meth:`.get_attacked`
        methods, which internally use MultiMaps for performance.
        """
        for attacker in self.attacked.keys():
            for attacked in self.attacked[attacker]:
                yield Attack(attacker=attacker, attacked=attacked)

    def add_argument(self, argument: Argument) -> Argument:
        """
        Add a new argument to the known arguments.

        :param argument: The argument to add. If an argument with the same
            name already exists, a warning is raised, and the new argument
            overrides the previous one. The argument is deep-copied before
            it is added, so that this AFDM can be modified without any
            side effect on other AFDMs, if they share arguments.

        :return: The new argument, which is a deep-copy of the given one.
        """
        if argument.name in self.A.keys():
            warnings.warn(f'Argument {argument.name} already exists!')
        new_argument = copy.deepcopy(argument)
        self.A[argument.name] = new_argument
        # Add (potential) decisions supported or countered by the argument
        # to the set of known decisions.
        self.decisions = self.decisions.union(argument.support)
        self.decisions = self.decisions.union(argument.counter)
        return new_argument

    def add_attack_relation(self,
                            attacker: Union[str, Argument],
                            attacked: Union[str, Argument]):
        """
        Add a new attack between arguments.

        :param attacker: The *attacker* argument, either identified by its
            name, or the argument itself.
        :param attacked: The *attacked* argument, either identified by its
            name, or the argument itself.

        :raises: An exception is raised if the arguments' names (either passed
            directly as parameters, or obtained through the arguments
            themselves) are not known, i.e., no argument is associated to
            them in this AFDM builder.
        """
        # We want the arguments' names (if we are passed Arguments directly).
        if isinstance(attacker, Argument):
            attacker = attacker.name
        if isinstance(attacked, Argument):
            attacked = attacked.name

        # Check that arguments are known, or at least that their names are known.
        if attacker not in self.A.keys():
            raise ArgumentNotFoundError(attacker)
        if attacked not in self.A.keys():
            raise ArgumentNotFoundError(attacked)

        # Add the attack relation
        self.attackers[attacked].add(attacker)
        self.attacked[attacker].add(attacked)

    def arguments_supporting(
            self,
            decision: str,
            ignore_aliveness: bool = False
    ) -> List[Argument]:
        """
        Return the arguments that support a given decision.

        This function corresponds to the ``F_f`` set in our algorithm, and is
        mostly used to get the number of supporting arguments, comparing it
        to the number of countering arguments, and resulting in a scalar
        reward. Optionally, returns all arguments, even those that are "killed"
        (not alive).

        :param decision: The decision that we are looking for. Typically,
            ``'moral'``.
        :param ignore_aliveness: Whether to return only *alive* arguments
            (by default), or to ignore the aliveness and return all arguments
            supporting ``decision`` (when set to ``True``).

        :return: The list of arguments that support ``decision``, i.e., that
            have ``decision`` as part of their :py:attr:`~.Argument.support`.
            When ``ignore_aliveness`` is ``False``, only arguments that have
            their :py:attr:`~.Argument.alive` set to ``True`` are returned.
            Otherwise, this condition is ignored.
        """
        result = []
        for argument in self.arguments:
            if (decision in argument.support) and \
                    (ignore_aliveness or argument.alive):
                result.append(argument)
        return result

    def arguments_countering(
            self,
            decision: str,
            ignore_aliveness: bool = False
     ) -> List[Argument]:
        """
        Return the arguments that counter a given decision.

        This function corresponds to the ``F_c`` set in our algorithm, and is
        mostly used to get the number of countering arguments, comparing it
        to the number of supporting arguments, and resulting in a scalar
        reward. Optionally, returns all arguments, even those that are "killed"
        (not alive).

        :param decision: The decision that we are looking for. Typically,
            ``'moral'``.
        :param ignore_aliveness: Whether to return only *alive* arguments
            (by default), or to ignore the aliveness and return all arguments
            supporting ``decision`` (when set to ``True``).

        :return: The list of arguments that counter ``decision``, i.e., that
            have ``decision`` as part of their :py:attr:`~.Argument.counter`.
            When ``ignore_aliveness`` is ``False``, only arguments that have
            their :py:attr:`~.Argument.alive` set to ``True`` are returned.
            Otherwise, this condition is ignored.
        """
        result = []
        for argument in self.arguments:
            if (decision in argument.counter) and \
                    (ignore_aliveness or argument.alive):
                result.append(argument)
        return result

    def is_alive(self, argument_name: str) -> bool:
        """
        Return whether an argument is alive in this AFDM.

        This is a shortcut for ``self.A[argument_name].alive``, which also
        makes the error more explicit if the argument cannot be found.

        :param argument_name: The name of the desired argument. Must be a valid
            key in :py:attr:`~.A`, otherwise an Exception is raised.

        :return: The argument's aliveness.
        """
        if argument_name not in self.A.keys():
            raise ArgumentNotFoundError(argument_name)
        return self.A[argument_name].alive

    def get_attackers(self,
                      argument: Union[Argument, str],
                      return_arguments=False) -> Union[List[str], List[Argument]]:
        """
        Return the arguments that are attackers of a given argument.

        :param argument: The desired argument, or its
            :py:attr:`~smartgrid.rewards.argumentation.lib.argument.Argument.name`.

        :param return_arguments: Whether to return the attackers as arguments
            directly, or to return their names.

        :return: The list of arguments that attack ``argument``, i.e., the
            arguments with name ``a`` such that ``('a', argument)`` are in
            :py:attr:`~.R`.
            When ``return_arguments`` is ``True``, the returned list is
            composed of instances of Arguments; otherwise, it is composed
            of the arguments' names (strings).
        """
        if isinstance(argument, Argument):
            argument = argument.name
        result = []
        for attacker in self.attackers[argument]:
            if return_arguments:
                result.append(self.A[attacker])
            else:
                result.append(attacker)
        return result

    def get_attacked(self,
                     argument: Union[Argument, str],
                     return_arguments=False) -> Union[List[str], List[Argument]]:
        """
        Return the arguments that are attacked by a given argument.

        :param argument: The desired argument, or its
            :py:attr:`~smartgrid.rewards.argumentation.lib.argument.Argument.name`.

        :param return_arguments: Whether to return the attacked as arguments
            directly, or to return their names.

        :return: The list of arguments that are attacked by ``argument``, i.e.,
            the arguments with name ``a`` such that ``(argument, 'a')`` are in
            :py:attr:`~.R`.
            When ``return_arguments`` is ``True``, the returned list is
            composed of instances of Arguments; otherwise, it is composed
            of the arguments' names (strings).
        """
        if isinstance(argument, Argument):
            argument = argument.name
        result = []
        for attacked in self.attacked[argument]:
            if return_arguments:
                result.append(self.A[attacked])
            else:
                result.append(attacked)
        return result

    def compute_grounded_extension(self) -> List[str]:
        """
        Compute the grounded extension.

        The grounded extension is one of the ways of computing the acceptable
        set of arguments, based on attacks between them.

        :return: The list of names of arguments that are in the grounded
            extension, i.e., which are considered acceptable.
        """
        # At the beginning, alive arguments are said to be `undecided`;
        # arguments that are already killed (because not true in this situation)
        # are `out`, so that they are not considered as suitable candidates
        # for the grounded extension.
        labels = {
            argument: 'undecided' if self.is_alive(argument) else 'out'
            for argument in self.A.keys()
        }

        # Take the next (candidate) node and iterate
        x = self._grounded_loop_condition(labels)
        while x is not None:
            # We "put" x in the grounded extension
            labels[x] = 'in'
            # All arguments attacked by x are therefore "killed"
            attacked = self.get_attacked(x)
            for z in attacked:
                labels[z] = 'out'
            # We take a new undecided node, for which all attackers are killed.
            x = self._grounded_loop_condition(labels)

        # We have now finished the grounded extension, there is no new candidate
        grounded = []
        for argument in labels.keys():
            if labels[argument] == 'in':
                grounded.append(argument)

        return grounded

    def _grounded_loop_condition(self, labels: Dict[str, str]) -> Optional[str]:
        """
        Find the next candidate for the grounded extension.

        Internal method that should not be useful externally.
        It finds an argument which is still ``undecided``, and for which all
        attackers have been killed (``out``), i.e., a suitable candidate for
        the grounded extension.

        :param labels: The arguments' labels when computing the grounded, i.e.,
            a dictionary mapping each argument name to a label (``undecided``,
            ``in``, or ``out``).

        :return: A new candidate argument, identified by its name, or ``None``
            if there is no remaining candidate (i.e., the grounded is complete).
        """
        for candidate in labels.keys():
            if labels[candidate] == 'undecided':
                valid = True
                attackers = self.get_attackers(candidate)
                for a in attackers:
                    if labels[a] != 'out':
                        valid = False
                        break
                if valid:
                    return candidate
        return None

    def update_alive_in_grounded_extension(self):
        """
        Compute the grounded extension and update the arguments' aliveness.

        The grounded extension is computed based on arguments that are alive or
        not, and attacks. Then, the arguments are said to be alive if they are
        in the grounded; otherwise, they are not.

        This method thus transitions from "activated arguments" (i.e., arguments
        that are true in the given situation) to "acceptable arguments" (i.e.,
        arguments that are true *and* defend each other).
        """
        self.grounded = self.compute_grounded_extension()
        for argument in self.arguments:
            # The argument is alive if it is in the grounded extension.
            # (The grounded cannot contain arguments that were not alive anyway).
            argument.alive = argument.name in self.grounded

    def __str__(self):
        return f'<AFDM #A={len(self.A)} ; #R={len(list(self.R))}>'

    __repr__ = __str__

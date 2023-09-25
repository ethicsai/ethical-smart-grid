"""
Defines an Attack between two Arguments.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Attack:
    """
    An attack between two arguments, represented as a tuple of names (strings).

    Attacks are implemented as dataclasses, instead of tuples, to avoid
    ambiguity. It is thus recommended to access the ``attacker`` and
    ``attacked`` fields by their name.


    """

    attacker: str
    """
    The name of the attacker Argument.
    """

    attacked: str
    """
    The name of the attacked Argument.
    """

    def __iter__(self) -> Iterable[str]:
        """
        Iterate over the Attack fields.

        Allows unpacking, like a tuple: ``(attacker, attacked) = attack``.
        """
        return iter((self.attacker, self.attacked))

    def short_str(self) -> str:
        """
        Return a short textual representation of an Attack.

        This short representation follows the format ``attacker -> attacked``,
        which is shorter than the traditional dataclass (``Attack(attacker=...,
        attacked=...)``).
        """
        return f'{self.attacker} -> {self.attacked}'

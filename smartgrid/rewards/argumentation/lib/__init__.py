"""
Internal library for argumentation graphs and judging agents.

This library defines the Argumentation Framework for Judging a Decision (AFJD),
based on the Argumentation Framework for Decision-Making (AFDM) by Amgoud et al.

This code is mostly based on the work of Benoît Alcaraz, refactored to simplify.

Amgoud, L., & Prade, H. (2009). Using arguments for making and explaining
decisions. Artificial Intelligence, 173(3-4), 413–436.
"""

from .afdm import AFDM
from .argument import Argument
from .attack import Attack
from .exceptions import ArgumentNotFoundError
from .judging_agent import JudgingAgent
from .judgment import (
    j_simple,
    j_diff,
    j_ratio,
    j_grad,
    j_offset,
)

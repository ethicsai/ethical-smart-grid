"""
The Q-SOM learning algorithm, leveraging 2 SOMs and a Q-Table.

The Self-Organizing Maps (SOMs) are used to handle the continuous and
multi-dimensional state and action spaces, whereas the Q-Table learns the
interests of actions in states.
"""

from .qsom import QSOM
from .qsom_agent import QsomAgent

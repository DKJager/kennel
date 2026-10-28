"""kennel — multi-model crowd answer engine.

Re-exports the public API: KennelError, chat, ask_crowd, reconcile, ground,
Votes, and lambda_handler.
"""

from kennel.client import KennelError, chat
from kennel.crowd import ask_crowd
from kennel.reconcile import reconcile
from kennel.ground import ground
from kennel.rsi import Votes
from kennel.handler import lambda_handler

__all__ = [
    "KennelError",
    "chat",
    "ask_crowd",
    "reconcile",
    "ground",
    "Votes",
    "lambda_handler",
]

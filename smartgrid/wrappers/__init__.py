"""
This module defines wrappers that can be used to change some functionalities
of the environment, such as reward aggregators that scalarize rewards, to
transform the multi-objective problem into a single-objective one.
"""

from .reward_aggregator import (RewardAggregator,
                                SingleRewardAggregator)

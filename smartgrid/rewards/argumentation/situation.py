"""
Helper function to parse situations (state-actions) for judgments.
"""

from typing import Dict, Any

from smartgrid import World
from smartgrid.agents import Agent


def parse_situation(world: World, agent: Agent) -> Dict[str, Any]:
    """
    Parse the current state-actions into a situation for judgment.

    :param world: The current world (equivalent to the state); also contains
        all other agents so that we can compare the currently judged agent
        to them.
    :param agent: The currently judged agent, mainly for its action, but
        may also refer to its profile, own personal state, etc.

    :return: A dict indexed by string keys, representing the current situation.
        The dict contains pre-processed elements that are easier to handle
        by the argumentation graphs.
    """
    s = {}
    comforts = [
        a.comfort for a in world.agents
    ]
    s['average_comfort'] = sum(comforts) / len(comforts)
    s['self_comfort'] = agent.comfort
    s['global_energy'] = world.available_energy
    s['agent_type'] = agent.profile.name
    given_energy = {
        a: a.enacted_action.give_energy
        for a in world.agents
    }
    s['average_give_energy'] = sum(given_energy.values()) / len(given_energy)
    s['give_energy'] = given_energy[agent]
    sorted_comforts = sorted(comforts)
    s['inter_quart_comfort'] = sorted_comforts[int(0.75 * len(comforts))] - \
                               sorted_comforts[int(0.25 * len(comforts))]
    s['min_max_comfort_diff'] = sorted_comforts[-1] - sorted_comforts[0]
    s['storage_n'] = agent.state.storage
    s['buy'] = agent.enacted_action.buy_energy
    s['payoff'] = agent.state.payoff
    s['self_need'] = agent.need
    s['grid_consumption'] = agent.enacted_action.grid_consumption
    s['storage_consumption'] = agent.enacted_action.storage_consumption
    s['sell'] = agent.enacted_action.sell_energy
    s['consumption'] = s['grid_consumption'] + s['storage_consumption']
    s['storage_n-1'] = agent.state.previous_storage
    s['store'] = agent.enacted_action.store_energy
    s['storage_capacity'] = agent.profile.max_storage
    s['solar'] = agent.state.production
    s['delta_store'] = s['store'] \
                       + s['buy'] \
                       + s['solar'] \
                       - s['sell'] \
                       - s['storage_consumption'] \
                       - s['give_energy']
    return s

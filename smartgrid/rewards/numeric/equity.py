from smartgrid.util.equity import hoover


def equity_reward(env, agent):
    """Reward based on the equity of comforts measure.

    It follows the principle of Difference Rewards: we compare the measure
    in the actual environment (in which the agent acted) and in an
    hypothetical environment (in which the agent would not have acted).
    If the actual environment is better than the hypothetical one, the
    agent's action improved it and should be rewarded.
    Otherwise, the agent degraded it and should be punished.
    """
    # Comforts of all other agents (excluding the current `agent`)
    other_comforts = [a.state.comfort for a in env.agents if a != agent]
    # Comfort of the current agent
    agent_comfort = agent.state.comfort

    # Compute the equity in the actual environment (others + agent)
    # we use 1-x since hoover returns 0=equity and 1=inequity
    actual_equity = 1.0 - hoover(other_comforts + [agent_comfort])

    # Compute the equity in the hypothetical environment
    hypothetical_equity = 1.0 - hoover(other_comforts)

    # Return the difference between the 2 environments
    return actual_equity - hypothetical_equity

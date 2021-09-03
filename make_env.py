"""
Code for creating a multi-agent environment with one of the scenarios listed
in ./scenarios/.

Can be called by using, for example:
    `env = make_env('simple_speaker_listener')`

The produced `env` object can be used similarly to an OpenAI gym environment.

A policy used on this environment must output actions in the form of a list
for all agents.
"""


def make_env(scenario_name):
    """
    Creates a SmartGrid object as env. This can be used similarly to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be used
                            (without the .py extension)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from smartgrid.environment import SmartGrid
    import smartgrid.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name).Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    env = SmartGrid(world)
    return env

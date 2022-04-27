from algorithms.qsom.qsom import QSOM
from smartgrid.scenarios.scenario import TestScenario

try:
    # Use `tqdm` to display a progress meter as we iterate over steps
    from tqdm import trange
except ImportError:
    # `tqdm` not available, simply use `range`
    trange = range


def run(config):
    env = make_env('debug')
    n_episodes = 1
    episode_length = 5

    obs = env.reset()
    for step in trange(episode_length):
        actions = [env.action_space[i].sample()
                   for i in range(len(env.action_space))]
        next_obs, rewards, _, infos = env.step(actions)
        obs = next_obs
    env.close()
    print('Done')


if __name__ == '__main__':
    run(None)

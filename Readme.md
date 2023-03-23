# Ethical Smart Grid Simulator

> Authors: Clément Scheirlinck, Rémy Chaput

## Description

This is a third-party [Gym] environment, focusing on learning ethically-aligned
behaviours in a Smart Grid use-case.

A Smart Grid contains several *prosumer* (prosumer-consumer) agents that
interact in a shared environment by consuming and exchanging energy.
These agents have an energy need, at each time step, that they must satisfy
by consuming energy. However, they should respect a set of moral values as
they do so, i.e., exhibiting an ethically-aligned behaviour.

Moral values are encoded in the reward functions, which determine the
"correctness" of an agent's action, with respect to these moral values.
Agents receive rewards as feedback that guide them towards a better behaviour.

## Quick usage

Clone this repository, open a Python shell (3.7+), and execute the following
instructions:

```python
from smartgrid import make_basic_smartgrid
from algorithms.qsom import QSOM

env = make_basic_smartgrid()
model = QSOM(env)

done = False
obs = env.reset()
while not done:
    actions = model.forward(obs)
    obs, rewards, terminated, truncated, _ = env.step(actions)
    model.backward(obs, rewards)
    done = all(terminated) or all(truncated)
env.close()
```

## License

The source code is licensed under the [MIT License].
Some included data may be protected by other licenses, please refer to the
[LICENSE.md] file for details.

[Gym]: https://gymnasium.farama.org/
[MIT License]: https://choosealicense.com/licenses/mit/
[LICENSE.md]: LICENSE.md

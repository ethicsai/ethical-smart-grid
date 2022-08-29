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

Clone this repository and execute the `main.py` entrypoint file:

```shell
python main.py --experiment="qsom_1" --model="QSOM" --scenario="ScenarioOne"
```

[Gym]: https://github.com/openai/gym

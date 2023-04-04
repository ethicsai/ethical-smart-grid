# Ethical Smart Grid Simulator

> Authors: Clément Scheirlinck, Rémy Chaput

<!-- Badges -->
![](https://img.shields.io/pypi/pyversions/ethical-smart-grid)
[![](https://img.shields.io/github/actions/workflow/status/ethicsai/ethical-smart-grid/docs.yml?label=Docs)](https://github.com/ethicsai/ethical-smart-grid/actions/workflows/docs.yml)
[![](https://img.shields.io/github/actions/workflow/status/ethicsai/ethical-smart-grid/testing.yml?label=Automatic%20testing)](https://github.com/ethicsai/ethical-smart-grid/actions/workflows/testing.yml)
![](https://img.shields.io/pypi/l/ethical-smart-grid)
![](https://img.shields.io/github/v/release/ethicsai/ethical-smart-grid)

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

## Versioning

This project follows the [Semver] (Semantic Versioning): all versions respect
the `<major>.<minor>.<patch>` format. The `patch` number is increased when a
bugfix is released. The `minor` number is increased when new features are added
that *do not* break the code public API, i.e., it is compatible with the
previous minor version. Finally, the `major` number is increased when a breaking
change is introduced; an important distinction is that such a change may not
be "important" in terms of lines of code, or number of features modified.
Simply changing a function's return type can be considered a breaking change
in the public API, and thus worthy of a "major" update.

## Building and testing locally

This GitHub repository includes actions that automatically [test][actions-test]
the package and [build][actions-docs] the documentation on each commit, and 
[publish][actions-publish] the package to [PyPi] on each release.

Instructions to perform these steps locally are given here, for potential
new contributors or forks:

- *Running the tests*

Tests are defined using [unittest] and run through [pytest]; please install it
first: `pip install pytest`.
We must add the current folder to the `PYTHONPATH` environment variable to
let pytest import the `smartgrid` module when executing the tests:
`export PYTHONPATH=$PWD` (from the root of this repository). Then, launch all
tests with `pytest tests`.

- *Building the documentation*

The documentation is built with [Sphinx] and requires additional requirements;
to install them, use `pip install -r docs/requirements.txt`. Then, to build the
documentation, use `cd docs && make html`. The built documentation will be in
the `docs/build/html` folder. It can be cleaned using `make clean` while in the
`docs` folder. Additionally, the `source/modules` folder is automatically
generated from the Python docstrings in the source code; it can be safely
deleted (e.g., with `rm -r source/modules`) to force re-building all
documentation files.

- *Building and publishing releases*

This project uses [hatch] to manage the building and publishing process; please
install it with `pip install hatch` first.

To build the package, use `hatch build` at the root of this repository. This
will create the *source distribution* (sdist) at
`dist/ethica_smart_grid_simulator-<version>.tar.gz`, and the *built distribution*
(wheel) at `dist/ethical_smart_grid_simulator-<version>-py3-none-any.whl`.

To publish these files to [PyPi], use `hatch publish`.

## License

The source code is licensed under the [MIT License].
Some included data may be protected by other licenses, please refer to the
[LICENSE.md] file for details.

[Gym]: https://gymnasium.farama.org/
[Semver]: https://semver.org/
[PyPi]: https://pypi.org/project/ethical-smart-grid/
[unittest]: https://docs.python.org/3/library/unittest.html
[pytest]: https://pytest.org/
[actions-test]: https://github.com/ethicsai/ethical-smart-grid/actions/workflows/testing.yml
[actions-docs]: https://github.com/ethicsai/ethical-smart-grid/actions/workflows/docs.yml
[actions-publish]: https://github.com/ethicsai/ethical-smart-grid/actions/workflows/package.yml
[Sphinx]: https://www.sphinx-doc.org/
[hatch]: https://hatch.pypa.io/latest/
[MIT License]: https://choosealicense.com/licenses/mit/
[LICENSE.md]: LICENSE.md

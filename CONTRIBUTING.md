# Contributing to ethical-smart-grid

Thank you for using this tool and considering to contribute!

All contributions are greatly appreciated; to make sure that these contributions
are efficiently managed, please follow the guidelines presented in this document.

*Note*: the project is not (currently) supported by a full-time dedicated person;
maintainers will try to answer as fast possible, depending on their availability.
We apologize for any delay.


## Getting support

If you find something unclear, or want to get help on how to achieve something,
you can ask questions by opening a new [Discussion under the Q&A category](https://github.com/ethicsai/ethical-smart-grid/discussions/new?category=q-a).

Please do *not* use issues to ask such questions, to avoid cluttering the
issue tracker: issues should be closed when treated, which makes them less
accessible by other members of the community, and thus lead to new (duplicate)
issues.
Instead, discussions will stay open and easily searchable by everyone.

Before creating a new discussion, please search the [existing questions](https://github.com/ethicsai/ethical-smart-grid/discussions/categories/q-a)
to avoid duplicates.

Everyone is free (and even encouraged!) to answer other questions.


## Reporting a bug

This software is tested against bugs, however some might have escaped our
attention.
If you find a bug, either an error raised by the Python interpreter, or an
unexpected result, please report it by creating a [new issue](https://github.com/ethicsai/ethical-smart-grid/issues/new),
using the *bug* label.

Similarly to the questions, please first search for [existing issues](https://github.com/ethicsai/ethical-smart-grid/issues/)
to avoid creating duplicates.
This will limit our workload, and ultimately allow us to answer you faster.

In the same vein, to facilitate our work, please *describe* as precisely as
possible the bug you encountered.
Detail what you installed, which code you executed before encountering the
bug, ...
When possible, a Minimal Working Example (MWE) is greatly appreciated.

Information about your Python environment can also be very important:
version of Python, Python shell or IPython, OS, installed packages, ...
An easy way to include several details is to copy the line printed by the
Python shell at launch, which should look like:

> Python 3.10.9 | packaged by conda-forge | (main, Feb  2 2023, 20:26:08) [Clang 14.0.6 ] on darwin

When describing an error, please also copy the full traceback emitted by Python.
All these details will help us pinpoint your problem quickly and easily, and,
in the end, to help you more efficiently.


## Proposing modifications

All changes, ranging from bug fixes to new features, as well as improvements to
the code and/or documentation, are also welcome!

The idea behind this project was to make it accessible to the community,
so that everyone can extend it, adapt it to their own liking, and make it a
useful tool.

### Small modifications

For small modifications, you may directly create a [Pull Request](https://github.com/ethicsai/ethical-smart-grid/pulls).
Please assign the correct label (*bug*, *documentation*, ...) to facilitate the
treatment of your PR.

Please make sure that the code works by running the following command, at the
root of the repository:

```shell
PYTHONPATH=$(PWD) pytest tests
```

For more details on how to build and test locally, please refer to the
[README](Readme.md#building-and-testing-locally).

### Larger modifications

To avoid "feature creep" (adding too much features), we would prefer to discuss
larger features first, by creating a new [Discussion under the Ideas category](https://github.com/ethicsai/ethical-smart-grid/discussions/new?category=ideas).

As a rule of thumb, features that break the existing API are considered
"large features" and should be discussed.
The whole community can participate in such discussions, to find the optimal
way to integrate these new features, in a way that is easily usable, while
maintaining an API as close as possible to the previous one, to avoid forcing
users to completely change the codebases that depend upon this project.

More largely, any idea, suggestion to improve (code, documentation, ...), or
request for a new feature, is also welcome for discussion.

---

We thank you again for using this software and considering to contribute.
Welcome to the community!

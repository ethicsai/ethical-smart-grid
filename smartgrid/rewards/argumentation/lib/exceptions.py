"""
Argumentation-specific exceptions.

We define specific classes here to simplify our job when raising Exceptions,
and to simplify the job of users when using try/except and identifying
which exception happened.
"""


class ArgumentNotFoundError(Exception):
    """
    Exception raised when an argument cannot be found.

    This exception is shown as ``Argument <name> not found!``, where ``<name>``
    is the requested name, which is accessible through the
    :py:attr:`~.ArgumentNotFoundError.name` attribute.
    """

    name: str
    """
    The requested name which does not correspond to any known argument.
    """

    def __init__(self, name):
        self.name = name
        self.message = f'Argument {name} not found!'

    def __str__(self):
        return self.message

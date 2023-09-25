"""
An Argument is the basic element of an argumentation graph.

This code is mostly based on the work of Benoît Alcaraz.
"""

import inspect
from typing import Callable, Any, List


class Argument:
    """
    An Argument is the basic element of an Argumentation Graph.

    Arguments represent something that may be true or false, and which are
    linked together by (attack) relationships.
    For example, "There are clouds in the sky" is an argument, which attacks
    "The weather is nice". If the "clouds" argument is true in a given
    situation, it will be difficult to accept the "nice weather" argument as
    well.
    """

    name: str
    """
    The name of an argument, a unique identifier.
    
    It should ideally be a string that intuitively gives a hint of the
    argument's content. In the above example "There are clouds in the sky",
    ``clouds``, ``cloudy``, or even ``cloudy_sky`` are reasonably good names,
    but ``arg1`` is not.
    """

    content: str
    """
    The content of an argument, a long description.
    
    This represents the argument itself, in a human-readable form.
    In the above example, "There are clouds in the sky" can be considered the
    content. ``"clouds in the sky"`` can be a shorter content, which still
    brings the same meaning.
    """

    activation_function: Callable[[Any], bool]
    """
    The activation function determines if an argument is alive in a situation.
    
    Arguments are defined outside of any context, but need to be evaluated
    within a given situation. For example, "clouds in the sky" is a possible
    argument, which may be true (*alive*) if there are effectively clouds; or
    false (*killed* or *disabled*) otherwise.
    
    An activation function takes a *situation* as parameter, which are left 
    untyped for flexibility: they usually will be dicts, but could be instances
    of specific classes, lists, dictionaries, ...
    The function must return a boolean, which indicates the argument's
    *aliveness* in the given situation.
    
    For example, assuming the situation is ``s = {'clouds': 4}`` (meaning that
    there are 4 clouds in the sky), an activation function for "clouds in the
    sky" could be ``lambda s: s['clouds'] > 0``.
    """

    alive: bool
    """
    Whether the argument is currently alive.
    
    The aliveness of an argument is determined by the :py:attr:`~activation_function`
    in a given situation.
    All arguments are alive by default, and should be updated when a situation
    is evaluated. They then should be reset after the judgment.
    """

    support: List[str]
    """
    The list of decisions this argument supports.
    
    This is a legacy from the AFDM, where the goal was to select a decision
    to make, supported by arguments. For example, "clouds in the sky" supports
    decision "take an umbrella", whereas "nice weather" supports "do not take
    an umbrella".
    
    In our case (AFJD), we simply want to judge whether the learning agent's
    action was aligned with a moral value; decisions can be simplified to
    ``'moral'`` (i.e., "yes, the action was aligned") and ``'immoral'`` (i.e.,
    "no, the action violates the moral value"), which is a binary choice.
    
    We keep the original definition, a list of decisions to support potential
    extensions, such as using different ethical principles to reason over
    moral values; yet, in practice, this list of decisions is currently limited.
    
    See also its counterpart, :py:attr:`~.counter`.
    """

    counter: List[str]
    """
    The list of decisions this argument counters.
    
    Similarly to :py:attr:`~.support`, this is a legacy from the AFDM.
    For example, "nice weather" counters decision "take an umbrella".
    
    Note that an argument can be neutral w.r.t. to a decision: it is not because
    it does not support it that it *must* counter it.
    For example, "clouds in the sky" does not necessarily counter "do not take
    an umbrella".
    """

    def __init__(self,
                 name: str,
                 content: str = "",
                 activation_function: Callable[[Any], bool] = None,
                 support: List[str] = None,
                 counter: List[str] = None,
                 ):
        """
        Create a new Argument.

        :param name: The name (unique identifier) of the argument.
        :param content: The (long) description of the argument.
        :param activation_function: The activation function of the argument,
            typically a ``lambda`` expression that takes a state as parameter,
            and returns a boolean. By default (``None``), the argument will
            always be considered activated (*alive*).
        :param support: The list of decisions that are supported by this
            argument. By default, an empty list.
        :param counter: The list of decisions that are countered by this
            argument. By default, an empty list.
        """
        self.name = name

        self.content = content

        if activation_function is None:
            activation_function = lambda s: True
        self.activation_function = activation_function

        self.alive = True

        if support is None:
            support = []
        self.support = support

        if counter is None:
            counter = []
        self.counter = counter

    def compute(self, state) -> bool:
        """
        Determine whether the argument is activated in a given state.

        :param state: The given state. Typically, a dict indexed by strings.

        :return: A boolean indicating whether the argument is activated.
        """
        return self.activation_function(state)

    def set_alive(self, alive: bool):
        """Change the argument's aliveness."""
        self.alive = alive

    def add_support(self, value: str):
        """Add a new value to the supports, if it is not present already."""
        # value = value.lower()
        if value not in self.support:
            self.support.append(value)

    def add_counter(self, value: str):
        """Add a new value to the counters, if it is not present already."""
        # value = value.lower()
        if value not in self.counter:
            self.counter.append(value)

    def __str__(self):
        """
        Short string representation of an Argument: contains its name.
        """
        return f'<Argument {self.name}>'

    def __repr__(self):
        """
        Long string representation of an Argument: contains all data.
        """
        # The `activation_function` can be a bit tricky to represent, as it is
        # code. We can use the `inspect` module, but we need to parse the result
        # a bit. It will be better than using `str` or `repr`, but not perfect...
        # In particular, there may be some left-over characters after the code,
        # such as `,` or `)`, depending on how and where the lambda is defined.
        # Yet, it can help identifying where is the function truly defined for
        # further debugging.
        try:
            code = inspect.getsourcelines(self.activation_function)
            # Source code is the first element of the tuple. We want the
            # first line of the source code.
            code = code[0][0]
            # Can be `def my_function(s):`, in which case we remove `def ` to
            # only get the function's name;
            # or `(...) lambda s: (...)`, with the first `(...)` being useless
            # here, and the second (...) being the actual code, in which case
            # we only want what is after the `lambda` keyword.
            pos = code.find('def')
            if pos != -1:
                # `def ` is in the code; let us skip these 4 characters
                code = code[pos+4:]
            pos = code.find('lambda')
            if pos != -1:
                # `lambda` is in the code; we want to retain only what is after
                code = code[pos:]
            # In any case, strip the spaces/newlines
            code = code.strip()
        except:
            # Something failed, let's not crash the app just for this, resort
            # to a more naïve string representation.
            code = str(self.activation_function)

        return f'<Argument name={self.name}; ' \
               f'content={self.content}; ' \
               f'activation_function={code}; ' \
               f'alive={self.alive}; ' \
               f'support={self.support}; ' \
               f'counter={self.counter}>'

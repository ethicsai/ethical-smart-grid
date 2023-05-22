"""
Argumentation-based reward functions.

Importing this package (or anything inside it) requires the :py:mod:`ajar`
library to be installed, and will raise an error otherwise.
"""

# Ensure that the user has installed AJAR with a custom (nicer) message,
# including how/where to install AJAR if necessary.
# Note: we should use `raise Exception(...) from e` or
# `Exception(...).with_traceback(e.__traceback__)`, however this over-complexifies
# the resulting error message.
try:
    import ajar
except ImportError as e:
    raise Exception(
        "Could not import 'ajar'.\n"
        "Please ensure you have installed the AJAR library with "
        "`pip install git+https://github.com/ethicsai/ajar.git@v1.0.0`\n"
        "Original error: " + e.msg
    )


from .affordability import Affordability
from .env_sustain import EnvironmentalSustainability
from .inclusiveness import Inclusiveness
from .supply_security import SupplySecurity

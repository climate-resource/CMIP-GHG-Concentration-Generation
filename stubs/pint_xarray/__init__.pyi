import pint

from . import accessors  # noqa: F401 # required to make stubs behave I think

def setup_registry(ur: pint.UnitRegistry) -> pint.UnitRegistry: ...

"""Utilities for pandas."""

from .core import FirstColFtClassifier  # noqa: F401
from .core import IdxBasedFtClassifier  # noqa: F401
from .core import FirstObjFtClassifier  # noqa: F401
from .core import ColLblBasedFtClassifier  # noqa: F401

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

for name in ['get_versions', '_version', 'core', 'name']:
    try:
        globals().pop(name)
    except KeyError:
        pass

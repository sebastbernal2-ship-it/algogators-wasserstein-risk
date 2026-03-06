"""
Algogators Wasserstein Risk Package

A toolkit for computing and analyzing Wasserstein risk metrics.
"""

__version__ = "0.1.0"

from . import config
from . import data
from . import features
from . import analysis

__all__ = [
    "config",
    "data",
    "features",
    "analysis",
]

"""Strategy plugin system for Flower FRL Benchmark."""

import importlib
import pkgutil

from frl_benchmark.strategies.base import (
    AggregationStrategy,
    register_strategy,
    get_strategy,
    list_strategies,
    apply_gradient,
)

__all__ = [
    "AggregationStrategy",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "apply_gradient",
]

# Auto-discover all strategy modules in this directory.
# Any .py file (except base.py) using @register_strategy is picked up automatically.
import frl_benchmark.strategies as _pkg
for _importer, _modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
    if _modname != "base":
        importlib.import_module(f"frl_benchmark.strategies.{_modname}")

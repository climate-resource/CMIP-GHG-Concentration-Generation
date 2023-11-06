"""
Miscellaneous pre-defined doit [TODO link] tasks
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from .typing import ConfigBundleLike, DoitTaskSpec

T = TypeVar("T")


def gen_show_config_tasks(
    config_bundle: ConfigBundleLike[T],
    display_func: Callable[[ConfigBundleLike[T]], None],
) -> Iterable[DoitTaskSpec]:
    """
    Generate tasks to show configuration

    Parameters
    ----------
    config_bundle
        Configuration bundle to display

    display_func
        Function to display (i.e. print) the configuration bundle

    Yields
    ------
        Task which displays the configuration (plus a base task which comes first)
    """
    yield {
        "name": config_bundle.run_id,
        "doc": "Show configuration that will be run",
        "actions": [(display_func, (config_bundle,), {})],
    }

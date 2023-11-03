"""
Miscellaneous pre-defined doit [TODO link] tasks
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from .typing import ConfigBundleLike, DoitTaskSpec

T = TypeVar("T")


def gen_show_config_tasks(
    config_bundles: Iterable[ConfigBundleLike[T]],
    display_func: Callable[[ConfigBundleLike[T]], None],
) -> Iterable[DoitTaskSpec]:
    """
    Generate tasks to show configuration

    Parameters
    ----------
    config_bundles
        Configuration bundles from which to show the configuration

    display_func
        Function to display (i.e. print) each configuration bundle

    Yields
    ------
        Task which displays a configuration (plus a base task which comes first)
    """
    base_task = {
        "name": None,
        "doc": "Show configurations to run",
    }
    yield {**base_task}

    for cb in config_bundles:
        yield {
            **base_task,
            "name": cb.config_id,
            "actions": [(display_func, (cb,), {})],
        }

"""
Tools for working with configuration
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def get_value_across_config_bundles(
    config_bundles: Iterable[T],
    access_func: Callable[[T], Any],
    expect_all_same: bool = False,
) -> list[Any] | Any:
    """
    Get a value from across multiple configuration bundles

    Parameters
    ----------
    config_bundles
        Configuration bundles to iterate through

    access_func
        Function that gets the desired value from each configuration bundle
    expect_all_same
        Should we expect a single, unique value to be returned?

    Returns
    -------
    list[Any] | Any
        Values from across the bundle. If ``expect_all_same`` is ``True`` then
        a single value is returned.

    Raises
    ------
    AssertionError:
        ``expect_all_same`` is ``True`` but the values aren't all the same.
    """
    vals = [access_func(cb) for cb in config_bundles]

    if expect_all_same:
        common_vals = set(vals)
        if len(common_vals) != 1:
            # TODO better error
            raise AssertionError(common_vals)

        return vals[0]

    return vals

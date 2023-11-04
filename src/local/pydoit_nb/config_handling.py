"""
Tools for working with configuration
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import attrs
import numpy as np
from attrs import evolve, fields

T = TypeVar("T")
U = TypeVar("U")


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


def insert_path_prefix(config: T, prefix: Path) -> T:
    """
    Insert path prefix into config attributes

    This adds the prefix ``prefix`` to any attributes of ``config`` which are
    :obj:`Path`

    Parameters
    ----------
    config
        Config to update

    prefix
        Prefix to add to paths

    Returns
    -------
        Updated ``config``
    """

    def update_attr_value(value: U) -> U:
        if isinstance(value, Path):
            return prefix / value

        if attrs.has(value):
            return insert_path_prefix(value, prefix)

        if not isinstance(value, str | np.ndarray) and isinstance(value, Iterable):
            return [update_attr_value(v) for v in value]

        return value

    config_attrs = fields(config.__class__)
    evolutions: dict[str, Any] = {}
    for attr in config_attrs:
        attr_name = attr.name
        attr_value = getattr(config, attr_name)

        evolutions[attr_name] = update_attr_value(attr_value)

    return evolve(config, **evolutions)

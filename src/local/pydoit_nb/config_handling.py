"""
Tools for working with configuration
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import attrs
import numpy as np
from attrs import evolve, fields

T = TypeVar("T")
U = TypeVar("U")


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
        """
        Update the attribute value if it is :obj:`Path` to include the prefix

        The prefix is taken from the outer scope

        Parameters
        ----------
        value
            Value to update

        Returns
        -------
            Updated value
        """
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


def get_branch_config_ids(configs: NotebookConfigLike) -> list[str]:
    return [c.branch_config_id for c in configs]

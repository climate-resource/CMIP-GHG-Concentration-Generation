"""
Tools for working with configuration
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, overload

import attrs
import numpy as np
from attrs import AttrsInstance, evolve, fields

from .typing import NotebookConfigLike

T = TypeVar("T")


def insert_path_prefix(config: AttrsInstance, prefix: Path) -> AttrsInstance:
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
    config_attrs = fields(config.__class__)

    evolutions: dict[str, Any] = {}
    for attr in config_attrs:
        attr_name = attr.name
        attr_value = getattr(config, attr_name)

        if attrs.has(type(attr_value)):
            evolutions[attr_name] = insert_path_prefix(attr_value, prefix)

        elif not isinstance(attr_value, str | np.ndarray) and isinstance(
            attr_value, Iterable
        ):
            evolutions[attr_name] = [update_attr_value(v, prefix) for v in attr_value]

        else:
            evolutions[attr_name] = update_attr_value(attr_value, prefix)

    return evolve(config, **evolutions)  # type: ignore # no idea why this fails


@overload
def update_attr_value(value: Path, prefix: Path) -> Path:
    ...


@overload
def update_attr_value(value: T, prefix: Path) -> T:
    ...


def update_attr_value(value: Path | T, prefix: Path) -> Path | T:
    """
    Update the attribute value if it is :obj:`Path` to include the prefix

    The prefix is taken from the outer scope

    Parameters
    ----------
    value
        Value to update

    prefix
        Prefix to insert before paths if ``value`` is an instance of ``Path``

    Returns
    -------
        Updated value
    """
    if isinstance(value, Path):
        return prefix / value

    return value


def get_branch_config_ids(configs: Iterable[NotebookConfigLike]) -> list[str]:
    return [c.branch_config_id for c in configs]

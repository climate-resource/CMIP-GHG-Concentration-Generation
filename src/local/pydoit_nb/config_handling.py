"""
Tools for working with configuration
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, cast, overload

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

        if not isinstance(attr_value, str | np.ndarray) and isinstance(
            attr_value, Iterable
        ):
            evolutions[attr_name] = [update_attr_value(v, prefix) for v in attr_value]

        else:
            evolutions[attr_name] = update_attr_value(attr_value, prefix)

    return evolve(config, **evolutions)  # type: ignore # no idea why this fails


@overload
def update_attr_value(value: AttrsInstance, prefix: Path) -> AttrsInstance:
    ...


@overload
def update_attr_value(value: Path, prefix: Path) -> Path:
    ...


@overload
def update_attr_value(value: T, prefix: Path) -> T:
    ...


def update_attr_value(
    value: AttrsInstance | Path | T, prefix: Path
) -> AttrsInstance | Path | T:
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
    if attrs.has(type(value)):
        return insert_path_prefix(cast(AttrsInstance, value), prefix)

    if isinstance(value, Path):
        return prefix / value

    return value


def get_branch_config_ids(configs: Iterable[NotebookConfigLike]) -> list[str]:
    """
    Get available config IDs from an iterable of notebook configurations

    Parameters
    ----------
    configs
        Configurations from which to retrieve the branch config IDs

    Returns
    -------
        Branch config ID from each config in ``configs``
    """
    return [c.branch_config_id for c in configs]


# TODO: fix this
get_step_config_ids = get_branch_config_ids

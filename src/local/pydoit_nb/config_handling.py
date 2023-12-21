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

        if isinstance(attr_value, dict):
            evolutions[attr_name] = {
                update_attr_value(k, prefix): update_attr_value(v, prefix)
                for k, v in attr_value.items()
            }

        elif not isinstance(attr_value, str | np.ndarray) and isinstance(
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


def get_step_config_ids(configs: Iterable[NotebookConfigLike]) -> list[str]:
    """
    Get available config IDs from an iterable of notebook configurations

    Parameters
    ----------
    configs
        Configurations from which to retrieve the step config IDs

    Returns
    -------
        Step config ID from each config in ``configs``
    """
    return [c.step_config_id for c in configs]


def get_config_for_step_id(
    config: Any,
    step: str,
    step_config_id: str,
) -> Any:
    """
    Get configuration for a specific value of step config ID for a specific step

    This will fail if ``step`` isn't a part of ``config``

    Parameters
    ----------
    config
        Config from which to retrieve the step config

    step
        Step from which to retrieve the configuration

    step_config_id
        The retrieved configuration's ``step_config_id`` will match this value

    Returns
    -------
        Configuration for step ``step`` with step config ID equal to
        ``step_config_id``

    Raises
    ------
    ValueError
        No configuration could be found with ID equal to ``step_config_id``
    """
    possibilities: Iterable[NotebookConfigLike] = getattr(config, step)
    available_step_config_ids = []
    for poss in possibilities:
        if poss.step_config_id == step_config_id:
            return poss
        else:
            available_step_config_ids.append(poss.step_config_id)

    raise ValueError(  # noqa: TRY003
        f"Couldn't find {step_config_id=}, available step config IDs: {available_step_config_ids}"
    )

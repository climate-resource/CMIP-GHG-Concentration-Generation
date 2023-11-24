"""
Configuration handling
"""
from __future__ import annotations

from functools import partial
from typing import Any, TypeAlias, cast

import cattrs.preconf.pyyaml
import numpy as np
import numpy.typing as npt

import local.pydoit_nb.serialization

from .base import Config

UnstructuredArray: TypeAlias = list[float] | list["UnstructuredArray"]


def unstructure_np_array(arr: npt.NDArray[np.float64]) -> UnstructuredArray:
    """
    Unstructure :obj:`npt.ArrayLike`

    This simply converts it to a list so is probably not very fast. However,
    this is just an example so could easily be optimised for production use if
    needed.

    Parameters
    ----------
    arr
        Array to unstructure

    Returns
    -------
        Unstructured array
    """
    return cast(UnstructuredArray, arr.tolist())


def structure_np_array(
    inp: UnstructuredArray, target_type: type[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Structure :obj:`npt.ArrayLke`

    The inverse of :func:`unstructure_np_array`

    Parameters
    ----------
    inp
        Data to structure

    target_type
        Type the data should be returned as

    Returns
    -------
        Structured array
    """
    return np.array(inp)


converter_yaml = cattrs.preconf.pyyaml.make_converter()


def _is_np_array(inp: Any) -> bool:
    return inp is np.ndarray or (getattr(inp, "__origin__", None) is np.ndarray)


converter_yaml.register_unstructure_hook_func(_is_np_array, unstructure_np_array)
converter_yaml.register_structure_hook_func(_is_np_array, structure_np_array)


load_config_from_file = partial(
    local.pydoit_nb.serialization.load_config_from_file,
    target=Config,
    converter=converter_yaml,
)


# TODO: move into pydoit_nb?
def get_config_for_step_id(
    config: Config,
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
    possibilities = getattr(config, step)
    for poss in possibilities:
        if poss.step_config_id == step_config_id:
            return poss

    raise ValueError(  # noqa: TRY003
        f"Couldn't find {step_config_id=}, available step config IDs: {possibilities}"
    )

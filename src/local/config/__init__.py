"""
Configuration handling
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, cast

import cattrs.preconf.pyyaml
import numpy as np
import numpy.typing as npt

from .base import Config, ConfigBundle

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

converter_yaml.register_unstructure_hook(npt.ArrayLike, unstructure_np_array)
converter_yaml.register_structure_hook(npt.ArrayLike, structure_np_array)


def get_config_bundles(
    root_dir_output: Path,
    run_id: str,
) -> list[ConfigBundle]:
    """
    Get configuration bundles

    All sorts of logic can be put in here. This is a very simple example.

    Parameters
    ----------
    root_dir_output
        Root directory in which output should be saved

    run_id
        ID for the run

    Returns
    -------
        Hydrated configuration bundles
    """
    configs = [
        Config(name="no-cov", covariance=np.array([[0.25, 0], [0, 0.5]])),
        Config(name="cov", covariance=np.array([[0.25, 0.5], [0.5, 0.5]])),
    ]

    bundles = [
        ConfigBundle(
            run_id=run_id,
            root_dir_output=root_dir_output,
            config_id=c.name,
            config_hydrated=c,
            config_hydrated_path=root_dir_output / run_id / c.name / f"{c.name}.yaml",
        )
        for c in configs
    ]

    return bundles

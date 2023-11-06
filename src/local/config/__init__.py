"""
Configuration handling
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, TypeAlias, cast

import cattrs.preconf.pyyaml
import numpy as np
import numpy.typing as npt

import local.pydoit_nb.serialization

from .base import Config, ConfigBundle

UnstructuredArray: TypeAlias = list[float] | list["UnstructuredArray"]

SEED: int = 28474038
"""Seed to use in random draws"""


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
    common_config = dict(seed=SEED)

    bundles = []
    for name, covariance in [
        ("no-cov", np.array([[0.25, 0], [0, 0.5]])),
        ("cov", np.array([[0.25, 0.5], [0.5, 0.5]])),
    ]:
        config_output_dir = root_dir_output / run_id / name
        config = Config(
            name=name,
            covariance=covariance,
            seed_file=root_dir_output / "seed.txt",
            **common_config,
        )
        bundle = ConfigBundle(
            run_id=run_id,
            root_dir_output=root_dir_output,
            output_notebook_dir=config_output_dir / "notebooks",
            config_id=config.name,
            config_hydrated=config,
            config_hydrated_path=config_output_dir / f"{config.name}.yaml",
        )

        bundles.append(bundle)

    return bundles


load_config_from_file = partial(
    local.pydoit_nb.serialization.load_config_from_file,
    target=Config,
    converter=converter_yaml,
)


def get_config_for_branch_id(
    config: Config,
    branch: str,
    branch_config_id: str,
) -> Any:
    possibilities = getattr(config, branch)
    for poss in possibilities:
        if poss.branch_config_id == branch_config_id:
            return poss

    raise AssertionError("Couldn't find config")

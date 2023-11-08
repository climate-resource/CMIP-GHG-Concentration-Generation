"""
Covariance plotting notebook steps
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from attrs import asdict

from ..config import get_config_for_branch_id
from ..pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from ..pydoit_nb.typing import ConfigBundleLike


def get_unconfigured_notebooks_covariance_plotting() -> Iterable[UnconfiguredNotebook]:
    """
    Get unconfigured notebooks for the covariance plotting branch

    Returns
    -------
        Unconfigured notebooks
    """
    return [
        UnconfiguredNotebook(
            notebook_path=Path("3xx_covariance-plotting") / "300_covariance-plotting",
            raw_notebook_ext=".py",
            summary="covariance_plotting - Quick plot to check covariance draws",
            doc="Quick plot to compare covariance draws. Complete plots come later",
        )
    ]


def configure_notebooks_covariance_plotting(
    unconfigured_notebooks: Iterable[UnconfiguredNotebook],
    config_bundle: ConfigBundleLike[Any],
    branch_name: str,
    branch_config_id: str,
) -> Iterable[ConfiguredNotebook]:
    """
    Configure notebooks for the covariance plotting branch

    Parameters
    ----------
    unconfigured_notebooks
        Unconfigured notebooks

    config_bundle
        Configuration bundle from which to take configuration values

    branch_name
        Name of the branch

    branch_config_id
        Branch config ID to use when configuring the notebook

    Returns
    -------
        Configured notebooks
    """
    uc_nbs_dict = {nb.notebook_path: nb for nb in unconfigured_notebooks}

    config = config_bundle.config_hydrated

    get_config_for_branch_id(
        config=config, branch=branch_name, branch_config_id=branch_config_id
    )

    config_covariance = config.covariance

    configured_notebooks = [
        ConfiguredNotebook(
            **asdict(
                uc_nbs_dict[Path("3xx_covariance-plotting") / "300_covariance-plotting"]
            ),
            configuration=None,
            dependencies=tuple([c.draw_file for c in config_covariance]),
            targets=(),
            config_file=config_bundle.config_hydrated_path,
            branch_config_id=branch_config_id,
        )
    ]

    return configured_notebooks

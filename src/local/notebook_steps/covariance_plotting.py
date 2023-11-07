"""
Covariance plotting notebook steps
"""
from __future__ import annotations

from pathlib import Path

from attrs import asdict

from ..config import get_config_for_branch_id

# TODO: move into pydoit_nb so it is more general?
from ..pydoit_nb.notebooks import ConfiguredNotebook, UnconfiguredNotebook


def get_unconfigured_notebooks_covariance_plotting():
    return [
        UnconfiguredNotebook(
            notebook_path=Path("3xx_covariance-plotting") / "300_covariance-plotting",
            raw_notebook_ext=".py",
            summary="covariance_plotting - Quick plot to check covariance draws",
            doc="Quick plot to compare covariance draws. Complete plots come later",
        )
    ]


def configure_notebooks_covariance_plotting(
    unconfigured_notebooks,
    config_bundle,
    branch_name,
    branch_config_id,
):
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
        )
    ]

    return configured_notebooks

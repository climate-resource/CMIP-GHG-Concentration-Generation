"""
Preparation notebook steps
"""
from __future__ import annotations

from pathlib import Path

from attrs import asdict

# TODO: move into pydoit_nb so it is more general?
from ..config import get_config_for_branch_id
from ..pydoit_nb.notebooks import ConfiguredNotebook, UnconfiguredNotebook


def get_unconfigured_notebooks_prep():
    return [
        UnconfiguredNotebook(
            notebook_path=Path("0xx_preparation") / "000_write-seed",
            raw_notebook_ext=".py",
            summary="prepare - write seed",
            doc="Write seed for random draws",
        )
    ]


def configure_notebooks_prep(
    unconfigured_notebooks,
    config_bundle,
    branch_name,
    branch_config_id,
):
    uc_nbs_dict = {nb.notebook_path: nb for nb in unconfigured_notebooks}

    config = config_bundle.config_hydrated

    config_branch = get_config_for_branch_id(
        config=config, branch=branch_name, branch_config_id=branch_config_id
    )

    configured_notebooks = [
        ConfiguredNotebook(
            **asdict(uc_nbs_dict[Path("0xx_preparation") / "000_write-seed"]),
            configuration=(config_branch.seed,),
            dependencies=(),
            targets=(config_branch.seed_file,),
            config_file=config_bundle.config_hydrated_path,
        )
    ]

    return configured_notebooks

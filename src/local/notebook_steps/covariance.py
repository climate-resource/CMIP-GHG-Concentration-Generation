"""
Covariance notebook steps
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from attrs import asdict

# TODO: move into pydoit_nb so it is more general?
from ..config import get_config_for_branch_id
from ..pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from ..pydoit_nb.typing import ConfigBundleLike


def get_unconfigured_notebooks_covariance() -> Iterable[UnconfiguredNotebook]:
    """
    Get unconfigured notebooks for the covariance branch

    Returns
    -------
        Unconfigured notebooks
    """
    return [
        UnconfiguredNotebook(
            notebook_path=Path("1xx_covariance") / "110_draw-samples",
            raw_notebook_ext=".py",
            summary="covariance - draw samples",
            doc="Draw samples with potential covariance",
        )
    ]


def configure_notebooks_covariance(
    unconfigured_notebooks: Iterable[UnconfiguredNotebook],
    config_bundle: ConfigBundleLike[Any],
    branch_name: str,
    branch_config_id: str,
) -> Iterable[ConfiguredNotebook]:
    """
    Configure notebooks for the covariance branch

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

    config_branch = get_config_for_branch_id(
        config=config, branch=branch_name, branch_config_id=branch_config_id
    )

    config_preparation = get_config_for_branch_id(
        config=config, branch="preparation", branch_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            **asdict(uc_nbs_dict[Path("1xx_covariance") / "110_draw-samples"]),
            configuration=(config_branch.covariance,),
            dependencies=(config_preparation.seed_file,),
            targets=(config_branch.draw_file,),
            config_file=config_bundle.config_hydrated_path,
            branch_config_id=branch_config_id,
        )
    ]

    return configured_notebooks

"""
Figures notebook steps
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


def get_unconfigured_notebooks_figures() -> Iterable[UnconfiguredNotebook]:
    """
    Get unconfigured notebooks for the figures branch

    Returns
    -------
        Unconfigured notebooks
    """
    return [
        UnconfiguredNotebook(
            notebook_path=Path("9xx_figures") / "910_create-clean-table",
            raw_notebook_ext=".py",
            summary="figures - Create clean table to plot from",
            doc="Create a clean data table from which to plot",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("9xx_figures") / "920_plot-draws",
            raw_notebook_ext=".py",
            summary="figures - Plot draws against each other",
            doc="Create a figure showing the different samples",
        ),
    ]


def configure_notebooks_figures(
    unconfigured_notebooks: Iterable[UnconfiguredNotebook],
    config_bundle: ConfigBundleLike[Any],
    branch_name: str,
    branch_config_id: str,
) -> Iterable[ConfiguredNotebook]:
    """
    Configure notebooks for the figures branch

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

    config_covariance = config.covariance
    config_constraint = config.constraint

    configured_notebooks = [
        ConfiguredNotebook(
            **asdict(uc_nbs_dict[Path("9xx_figures") / "910_create-clean-table"]),
            configuration=(),
            dependencies=tuple(
                [c.draw_file for c in config_covariance]
                + [c.draw_file for c in config_constraint]
            ),
            targets=(config_branch.draw_comparison_table,),
            config_file=config_bundle.config_hydrated_path,
            branch_config_id=branch_config_id,
        ),
        ConfiguredNotebook(
            **asdict(uc_nbs_dict[Path("9xx_figures") / "920_plot-draws"]),
            configuration=(),
            dependencies=(config_branch.draw_comparison_table,),
            targets=(config_branch.draw_comparison_figure,),
            config_file=config_bundle.config_hydrated_path,
            branch_config_id=branch_config_id,
        ),
    ]

    return configured_notebooks

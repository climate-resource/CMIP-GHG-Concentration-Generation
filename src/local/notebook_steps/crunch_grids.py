"""
Crunch grids notebook steps
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydoit_nb.config_handling import get_config_for_step_id
from pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from pydoit_nb.notebook_step import UnconfiguredNotebookBasedStep

if TYPE_CHECKING:
    from ..config.base import Config, ConfigBundle


def configure_notebooks(
    unconfigured_notebooks: Iterable[UnconfiguredNotebook],
    config_bundle: ConfigBundle,
    step_name: str,
    step_config_id: str,
) -> list[ConfiguredNotebook]:
    """
    Configure notebooks

    Parameters
    ----------
    unconfigured_notebooks
        Unconfigured notebooks

    config_bundle
        Configuration bundle from which to take configuration values

    step_name
        Name of the step

    step_config_id
        Step config ID to use when configuring the notebook

    Returns
    -------
        Configured notebooks
    """
    uc_nbs_dict = {nb.notebook_path: nb for nb in unconfigured_notebooks}

    config = config_bundle.config_hydrated

    config_step = get_config_for_step_id(
        config=config, step=step_name, step_config_id=step_config_id
    )

    if config_step.gas in ("co2", "ch4", "n2o"):
        step = f"calculate_{config_step.gas}_monthly_fifteen_degree_pieces"
        step_config_id_gridding_pieces_step = "only"

    elif config_step.gas in (
        "c4f10",
        "c5f12",
        "c6f14",
        "c7f16",
        "c8f18",
    ):
        step = "calculate_c4f10_like_monthly_fifteen_degree_pieces"
        step_config_id_gridding_pieces_step = config_step.gas

    else:
        step = "calculate_sf6_like_monthly_fifteen_degree_pieces"
        step_config_id_gridding_pieces_step = config_step.gas

    config_gridding_pieces_step = get_config_for_step_id(
        config=config,
        step=step,
        step_config_id=step_config_id_gridding_pieces_step,
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[Path("30yy_grid") / "3001_crunch-grids"],
            configuration=(),
            dependencies=(
                config_gridding_pieces_step.global_annual_mean_allyears_monthly_file,
                config_gridding_pieces_step.seasonality_allyears_fifteen_degree_monthly_file,
                config_gridding_pieces_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file,
            ),
            targets=(
                config_step.fifteen_degree_monthly_file,
                # config_step.half_degree_monthly_file,
                config_step.gmnhsh_mean_monthly_file,
                config_step.gmnhsh_mean_annual_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="crunch_grids",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("30yy_grid") / "3001_crunch-grids",
            raw_notebook_ext=".py",
            summary="grid - Grid data from the gridding pieces",
            doc=(
                "Create gridded data products based on the seasonality, "
                "latitutindal gradient and global-means from earlier steps"
            ),
        ),
    ],
    configure_notebooks=configure_notebooks,
)

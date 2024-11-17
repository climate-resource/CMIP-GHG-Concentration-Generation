"""
Calculate C4F10 gases monthly 15 degree pieces notebook steps
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

    config_step = get_config_for_step_id(config=config, step=step_name, step_config_id=step_config_id)

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("14yy_c4f10-like-monthly-15-degree") / "1405_c4f10-like_create-pieces-for-gridding"
            ],
            configuration=(),
            dependencies=(),
            targets=(
                config_step.global_annual_mean_allyears_monthly_file,
                config_step.seasonality_allyears_fifteen_degree_monthly_file,
                config_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="calculate_c4f10_like_monthly_fifteen_degree_pieces",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("14yy_c4f10-like-monthly-15-degree")
            / "1405_c4f10-like_create-pieces-for-gridding",
            raw_notebook_ext=".py",
            summary="C4F10-like gas pieces - Finalise the pieces for gridding",
            doc="Finalise the pieces required for creating gridded files",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

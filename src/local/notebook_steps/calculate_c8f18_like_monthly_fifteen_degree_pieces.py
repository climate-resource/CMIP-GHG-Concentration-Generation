"""
Calculate C8F18 gases monthly 15 degree pieces notebook steps
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

    config_historical_emissions = get_config_for_step_id(
        config=config, step="compile_historical_emissions", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("15yy_c8f18-like-monthly-15-degree") / "1505_c8f18-like_create-pieces-for-gridding"
            ],
            configuration=(),
            dependencies=(config_historical_emissions.complete_historical_emissions_file,),
            targets=(
                # TODO: save out regression between lat. gradient and emissions too
                # Save out the historical emissions we used for that regression too,
                # to support harmonisation
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
    step_name="calculate_c8f18_like_monthly_fifteen_degree_pieces",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("15yy_c8f18-like-monthly-15-degree")
            / "1505_c8f18-like_create-pieces-for-gridding",
            raw_notebook_ext=".py",
            summary="c8f18-like gas pieces - Create the pieces for gridding",
            doc="Create the pieces required for creating gridded files from CMIP6 data",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

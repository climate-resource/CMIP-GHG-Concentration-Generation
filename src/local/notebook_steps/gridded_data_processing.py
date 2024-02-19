"""
Gridded data processing notebook steps
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
    config_grid = get_config_for_step_id(
        config=config, step="grid", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("08yy_gridded-data-processing") / "0810_create-annual-global-mean"
            ],
            configuration=None,
            dependencies=(config_grid.processed_data_file,),
            targets=(
                config_step.processed_data_file_global_hemispheric_means,
                config_step.processed_data_file_global_hemispheric_annual_means,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("08yy_gridded-data-processing") / "0820_downscale-to-fine-grid"
            ],
            configuration=None,
            dependencies=(config_grid.processed_data_file,),
            targets=(),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="gridded_data_processing",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("08yy_gridded-data-processing")
            / "0810_create-annual-global-mean",
            raw_notebook_ext=".py",
            summary="gridded data processing - global-, hemispheric- and annual-mean",
            doc="Create global-, hemispheric- and annual-means from gridded data",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("08yy_gridded-data-processing")
            / "0820_downscale-to-fine-grid",
            raw_notebook_ext=".py",
            summary="gridded data processing - downscale to fine grid",
            doc="Downscale to finer grid from gridded data",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

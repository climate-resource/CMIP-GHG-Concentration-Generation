"""
Process NOAA HATS data
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

    config_retrieve_misc_data = get_config_for_step_id(
        config=config, step="retrieve_misc_data", step_config_id="only"
    )
    config_retrieve_noaa = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_noaa_data",
        step_config_id=f"{config_step.gas}_hats",
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[Path("001y_process-noaa-data") / "0014_process_hats"],
            configuration=(),
            dependencies=(
                config_retrieve_noaa.interim_files["monthly_data"],
                (
                    config_retrieve_misc_data.natural_earth.raw_dir
                    / config_retrieve_misc_data.natural_earth.countries_shape_file_name
                ),
            ),
            targets=(config_step.processed_monthly_data_with_loc_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="process_noaa_hats_data",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("001y_process-noaa-data") / "0014_process_hats",
            raw_notebook_ext=".py",
            summary="process NOAA HATS data - process",
            doc=(
                "Process NOAA HATS data to create a file with monthly average "
                "from each station and latitude and longitude information"
            ),
        ),
    ],
    configure_notebooks=configure_notebooks,
)

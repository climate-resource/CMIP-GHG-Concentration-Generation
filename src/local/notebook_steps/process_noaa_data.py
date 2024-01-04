"""
Process NOAA data notebook steps
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from ..pydoit_nb.config_handling import get_config_for_step_id
from ..pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from ..pydoit_nb.notebook_step import UnconfiguredNotebookBasedStep

if TYPE_CHECKING:
    from ..config.base import ConfigBundle


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
    config_retrieve = get_config_for_step_id(
        config=config, step="retrieve", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("001y_process-noaa-data") / "0010_download"
            ],
            configuration=(
                config_step.download_urls,
                config_step.raw_dir,
            ),
            dependencies=(),
            targets=(config_step.download_complete_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("001y_process-noaa-data") / "0011_extract"
            ],
            configuration=(),
            dependencies=(config_step.download_complete_file,),
            targets=(
                config_step.interim_files["events_data"],
                config_step.interim_files["monthly_data"],
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("001y_process-noaa-data") / "0012_process"
            ],
            configuration=(),
            dependencies=(
                config_step.interim_files["events_data"],
                config_step.interim_files["monthly_data"],
                (
                    config_retrieve.natural_earth.raw_dir
                    / config_retrieve.natural_earth.countries_shape_file_name
                ),
            ),
            targets=(config_step.processed_monthly_data_with_loc_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step = UnconfiguredNotebookBasedStep(
    step_name="process_noaa_data",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("001y_process-noaa-data") / "0010_download",
            raw_notebook_ext=".py",
            summary="process NOAA data - download",
            doc="Download NOAA data",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("001y_process-noaa-data") / "0011_extract",
            raw_notebook_ext=".py",
            summary="process NOAA data - extract",
            doc="Extract NOAA data from zip files into interim formats",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("001y_process-noaa-data") / "0012_process",
            raw_notebook_ext=".py",
            summary="process NOAA data - process",
            doc=(
                "Process NOAA data to create a file with monthly average "
                "from each station and latitude and longitude information"
            ),
        ),
    ],
    configure_notebooks=configure_notebooks,
)

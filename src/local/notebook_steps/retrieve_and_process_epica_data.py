"""
Retrieve and extract EPICA data notebook steps
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydoit_nb.checklist import get_checklist_file
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

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("005y_process-epica-data") / "0050_download-epica"
            ],
            configuration=(config_step.download_url,),
            dependencies=(),
            targets=(get_checklist_file(config_step.raw_dir),),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("005y_process-epica-data") / "0051_process-epica"
            ],
            configuration=(),
            dependencies=(get_checklist_file(config_step.raw_dir),),
            targets=(config_step.processed_data_with_loc_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="retrieve_and_process_epica_data",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("005y_process-epica-data") / "0050_download-epica",
            raw_notebook_ext=".py",
            summary="process EPICA data - download",
            doc="Download EPICA data",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("005y_process-epica-data") / "0051_process-epica",
            raw_notebook_ext=".py",
            summary="process EPICA data - process",
            doc="Process EPICA data into a single file with monthly information etc.",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

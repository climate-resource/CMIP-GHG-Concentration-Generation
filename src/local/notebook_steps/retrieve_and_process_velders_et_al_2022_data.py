"""
Retrieve and process Velders et al. (2022) data notebook steps
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
        # Turn back on when Guus has updated zenodo
        # ConfiguredNotebook(
        #     unconfigured_notebook=uc_nbs_dict[
        #         Path("009y_process-velders-et-al-2022-data") / "0091_download-velders-et-al-2022-data"
        #     ],
        #     configuration=(config_step.zenodo_record,),
        #     dependencies=(),
        #     targets=(config_step.download_complete_file,),
        #     config_file=config_bundle.config_hydrated_path,
        #     step_config_id=step_config_id,
        # ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("009y_process-velders-et-al-2022-data") / "0092_process-velders-et-al-2022-data"
            ],
            configuration=(config_step.raw_data_file_tmp,),
            dependencies=(),
            targets=(config_step.processed_data_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="retrieve_and_process_velders_et_al_2022_data",
    unconfigured_notebooks=[
        # Turn back on when Guus has updated zenodo
        # UnconfiguredNotebook(
        #     notebook_path=Path("009y_process-velders-et-al-2022-data")
        #     / "0091_download-velders-et-al-2022-data",
        #     raw_notebook_ext=".py",
        #     summary="Retrieve and process Velders et al. (2022) data - download",
        #     doc="Download Velders et al. (2022) data",
        # ),
        UnconfiguredNotebook(
            notebook_path=Path("009y_process-velders-et-al-2022-data")
            / "0092_process-velders-et-al-2022-data",
            raw_notebook_ext=".py",
            summary="Retrieve and process Velders et al. (2022) data - process",
            doc="Process Velders et al. (2022) data",
        ),
    ],
    configure_notebooks=configure_notebooks,
)
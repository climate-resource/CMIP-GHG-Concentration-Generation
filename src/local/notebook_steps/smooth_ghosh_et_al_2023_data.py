"""
Ghosh et al. 2023 data smoothing notebook steps
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
    config_process_ghosh_et_al_2023 = get_config_for_step_id(
        config=config, step="retrieve_and_process_ghosh_et_al_2023_data", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("031y_smooth-ghosh-et-al-2023-data") / "0311_smooth-ghosh-et-al-2023-data"
            ],
            configuration=(),
            dependencies=(config_process_ghosh_et_al_2023.processed_data_file,),
            targets=(config_step.smoothed_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        )
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="smooth_ghosh_et_al_2023_data",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("031y_smooth-ghosh-et-al-2023-data") / "0311_smooth-ghosh-et-al-2023-data",
            raw_notebook_ext=".py",
            summary="Smooth Ghosh et al. 2023 data - do smoothing",
            doc=("Smooth the Ghosh et al. 2023 data."),
        )
    ],
    configure_notebooks=configure_notebooks,
)

"""
Law Dome data smoothing notebook steps
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
    config_process_law_dome = get_config_for_step_id(
        config=config, step="retrieve_and_process_law_dome_data", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("030y_smooth-law-dome-data") / "0301_smooth-law-dome-data"
            ],
            configuration=(
                config_step.n_draws,
                config_step.noise_adder,
                config_step.point_selector_settings,
            ),
            dependencies=(config_process_law_dome.processed_data_with_loc_file,),
            targets=(config_step.smoothed_draws_file, config_step.smoothed_median_file),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        )
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="smooth_law_dome_data",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("030y_smooth-law-dome-data") / "0301_smooth-law-dome-data",
            raw_notebook_ext=".py",
            summary="Smooth Law Dome data - do smoothing",
            doc=(
                "Add uncertainty to the Law Dome data, "
                "then smooth it using a weighted quantile regression. "
                "Repeat multiple times then take the median of the result."
            ),
        )
    ],
    configure_notebooks=configure_notebooks,
)

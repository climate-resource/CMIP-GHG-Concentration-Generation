"""
Write files in input4MIPs format notebook steps
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

    if "eq" in config_step.gas:
        config_crunch_grids = get_config_for_step_id(
            config=config,
            step="crunch_equivalent_species",
            step_config_id=config_step.gas,
        )

    else:
        config_crunch_grids = get_config_for_step_id(
            config=config,
            step="crunch_grids",
            step_config_id=config_step.gas,
        )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("40yy_write-input4mips") / "4001_write-input4mips-files"
            ],
            configuration=(config.doi,),
            dependencies=(
                config_crunch_grids.fifteen_degree_monthly_file,
                # config_crunch_grids.half_degree_monthly_file,
                config_crunch_grids.global_mean_monthly_file,
                config_crunch_grids.hemispheric_mean_monthly_file,
                config_crunch_grids.global_mean_annual_mean_file,
                config_crunch_grids.hemispheric_mean_annual_mean_file,
            ),
            targets=(
                # get_checklist_file(config_step.input4mips_out_dir),
                config_step.complete_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="write_input4mips",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("40yy_write-input4mips") / "4001_write-input4mips-files",
            raw_notebook_ext=".py",
            summary="write input4MIPs - write all files",
            doc="Write input4MIPs files for our four data products",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

"""
Process raw data notebook steps
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
    config_retrieve = get_config_for_step_id(
        config=config, step="retrieve", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("01yy_process-data") / "0101_process-law-dome"
            ],
            configuration=(config_step.law_dome,),
            dependencies=(get_checklist_file(config_retrieve.law_dome.raw_dir),),
            targets=(config_step.law_dome.processed_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("01yy_process-data") / "0111_process-gggrn-global-mean"
            ],
            configuration=None,
            dependencies=(get_checklist_file(config_retrieve.gggrn.raw_dir),),
            targets=(config_step.gggrn.processed_file_global_mean,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="process",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("01yy_process-data") / "0101_process-law-dome",
            raw_notebook_ext=".py",
            summary="process - Law Dome",
            doc="Process data for Law Dome observations",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("01yy_process-data") / "0111_process-gggrn-global-mean",
            raw_notebook_ext=".py",
            summary="process - Global Greenhouse Gas Research Network (GGGRN)",
            doc=(
                "Process data from the Global Greenhouse Gas Research Network (GGGRN). "
                "At present, this notebook only processes global-mean data."
            ),
        ),
    ],
    configure_notebooks=configure_notebooks,
)

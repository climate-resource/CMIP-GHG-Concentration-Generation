"""
Quick crunch notebook steps
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
    config_process = get_config_for_step_id(
        config=config, step="process", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("0aaa_quick-crunch") / "yyyy_quick-crunch-global-mean"
            ],
            configuration=None,
            dependencies=(
                config_process.gggrn.processed_file_global_mean,
                config_process.law_dome.processed_file,
            ),
            targets=(config_step.processed_data_file_global_means,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        )
    ]

    return configured_notebooks


step = UnconfiguredNotebookBasedStep(
    step_name="quick_crunch",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("0aaa_quick-crunch") / "yyyy_quick-crunch-global-mean",
            raw_notebook_ext=".py",
            summary="quick crunch - Create combined global-mean",
            doc=(
                "Create combined global-mean. This is just a placeholder until "
                "we implemented a more sophisticated routine"
            ),
        )
    ],
    configure_notebooks=configure_notebooks,
)

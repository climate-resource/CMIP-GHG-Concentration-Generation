"""
Covariance plotting notebook steps
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

    get_config_for_step_id(config=config, step=step_name, step_config_id=step_config_id)

    config_covariance = config.covariance

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("3xx_covariance-plotting") / "300_covariance-plotting"
            ],
            configuration=None,
            dependencies=tuple([c.draw_file for c in config_covariance]),
            targets=(),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        )
    ]

    return configured_notebooks


step = UnconfiguredNotebookBasedStep(
    step_name="covariance_plotting",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("3xx_covariance-plotting") / "300_covariance-plotting",
            raw_notebook_ext=".py",
            summary="covariance_plotting - Quick plot to check covariance draws",
            doc="Quick plot to compare covariance draws. Complete plots come later",
        )
    ],
    configure_notebooks=configure_notebooks,
)

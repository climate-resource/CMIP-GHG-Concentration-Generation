"""
Retrieve notebook steps
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
                Path("00yy_retrieve-data") / "0001_law-dome"
            ],
            configuration=(config_step.law_dome,),
            dependencies=(),
            targets=(get_checklist_file(config_step.law_dome.raw_dir),),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("00yy_retrieve-data") / "0011_gggrn"
            ],
            configuration=(config_step.gggrn.urls_global_mean,),
            dependencies=(),
            targets=(get_checklist_file(config_step.gggrn.raw_dir),),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("000y_retrieve-misc-data") / "0001_natural-earth-shape-files"
            ],
            configuration=(config_step.natural_earth.download_urls,),
            dependencies=(),
            targets=(
                (
                    config_step.natural_earth.raw_dir
                    / config_step.natural_earth.countries_shape_file_name
                ),
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="retrieve",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("00yy_retrieve-data") / "0001_law-dome",
            raw_notebook_ext=".py",
            summary="retrieve - Law Dome",
            doc="Retrieve data for Law Dome observations",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("00yy_retrieve-data") / "0011_gggrn",
            raw_notebook_ext=".py",
            summary="retrieve - Global Greenhouse Gas Research Network (GGGRN)",
            doc=(
                "Retrieve data from the Global Greenhouse Gas Research Network (GGGRN). "
                "At present, this notebook only retrieves global-mean data."
            ),
        ),
        UnconfiguredNotebook(
            notebook_path=Path("000y_retrieve-misc-data")
            / "0001_natural-earth-shape-files",
            raw_notebook_ext=".py",
            summary="retrieve - Natural Earth shape files",
            doc="Retrieve shape files from Natural Earth",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

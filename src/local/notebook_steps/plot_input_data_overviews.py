"""
Plot notebook steps
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydoit_nb.config_handling import get_config_for_step_id  # noqa: F401
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

    # config_step = get_config_for_step_id(config=config, step=step_name, step_config_id=step_config_id)

    # Multiple loops because mypy is being stupid
    dependencies_noaa = []
    for c_noaa_surface_flask in config.process_noaa_surface_flask_data:
        dependencies_noaa.append(
            c_noaa_surface_flask.processed_monthly_data_with_loc_file
        )

    for c_noaa_in_situ in config.process_noaa_in_situ_data:
        dependencies_noaa.append(c_noaa_in_situ.processed_monthly_data_with_loc_file)

    dependencies_agage = []
    for c_ale in config.retrieve_and_extract_ale_data:
        dependencies_agage.append(c_ale.processed_monthly_data_with_loc_file)

    for c_gage in config.retrieve_and_extract_gage_data:
        dependencies_agage.append(c_gage.processed_monthly_data_with_loc_file)

    for c_agage in config.retrieve_and_extract_agage_data:
        dependencies_agage.append(c_agage.processed_monthly_data_with_loc_file)

    dependencies_law_dome = tuple(
        c.processed_data_with_loc_file
        for c in config.retrieve_and_process_law_dome_data
    )
    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("001y_process-noaa-data") / "0019_noaa-network-overview"
            ],
            configuration=(),
            dependencies=tuple(dependencies_noaa),
            targets=(),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("002y_process-agage-data") / "0029_agage-network-overview"
            ],
            configuration=(),
            dependencies=tuple(dependencies_agage),
            targets=(),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("003y_process-law-dome-data") / "0039_plot-overview-law-dome"
            ],
            configuration=(),
            dependencies=dependencies_law_dome,
            targets=(),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="plot_input_data_overviews",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("001y_process-noaa-data") / "0019_noaa-network-overview",
            raw_notebook_ext=".py",
            summary="plot - Plot NOAA network overview",
            doc="Plot an overview of the NOAA network for all gases",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("002y_process-agage-data")
            / "0029_agage-network-overview",
            raw_notebook_ext=".py",
            summary="plot - Plot AGAGE network overview",
            doc="Plot an overview of the AGAGE network for all gases",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("003y_process-law-dome-data")
            / "0039_plot-overview-law-dome",
            raw_notebook_ext=".py",
            summary="plot - Plot Law Dome observations overview",
            doc="Plot an overview of the Law Dome observations for all gases",
        ),
    ],
    configure_notebooks=configure_notebooks,
)

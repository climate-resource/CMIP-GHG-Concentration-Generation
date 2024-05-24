"""
Crunch equivalent species notebook steps
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

    config_grid_crunching_included_gases = [
        get_config_for_step_id(
            config=config,
            step="crunch_grids",
            step_config_id=gas,
        )
        for gas in config_step.equivalent_component_gases
    ]

    dependencies = tuple(
        [
            getattr(crunch_gas_config, attr_to_grab)
            for crunch_gas_config in config_grid_crunching_included_gases
            for attr_to_grab in (
                "fifteen_degree_monthly_file",
                # "half_degree_monthly_file",
                "gmnhsh_mean_monthly_file",
                "gmnhsh_mean_annual_file",
            )
        ]
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("35yy_equivalent-species") / "3501_calculate-full-equivalence"
            ],
            configuration=(),
            dependencies=dependencies,
            targets=(
                config_step.fifteen_degree_monthly_file,
                # config_step.half_degree_monthly_file,
                config_step.gmnhsh_mean_monthly_file,
                config_step.gmnhsh_mean_annual_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="crunch_equivalent_species",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("35yy_equivalent-species")
            / "3501_calculate-full-equivalence",
            raw_notebook_ext=".py",
            summary="equivalent species - Calculate equivalent species timeseries",
            doc=(
                "Create full equivalent species. "
                "These allow modelling groups to have fewer tracers but the same radiative effect."
            ),
        ),
    ],
    configure_notebooks=configure_notebooks,
)

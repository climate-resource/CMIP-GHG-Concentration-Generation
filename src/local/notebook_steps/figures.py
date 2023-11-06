"""
Figures notebook steps
"""
from __future__ import annotations

from pathlib import Path

# TODO: move into pydoit_nb so it is more general?
from ..config import get_config_for_branch_id
from ..pydoit_nb.notebook_step import NotebookStep
from ..pydoit_nb.notebooks import UnconfiguredNotebook


def get_figures_notebook_steps(
    config_bundle,
    branch_name: str,
    branch_config_id: str,
    root_dir_raw_notebooks: Path,
):
    config = config_bundle.config_hydrated

    config_branch = get_config_for_branch_id(
        config=config, branch=branch_name, branch_config_id=branch_config_id
    )

    config_covariance = config.covariance
    config_constraint = config.constraint

    figures_notebooks = [
        UnconfiguredNotebook(
            notebook_path=Path("9xx_figures") / "910_create-clean-table",
            raw_notebook_ext=".py",
            summary="figures - Create clean table to plot from",
            doc="Create a clean data table from which to plot",
            configuration=(),
            dependencies=tuple(
                [c.draw_file for c in config_covariance]
                + [c.draw_file for c in config_constraint]
            ),
            targets=(config_branch.draw_comparison_table,),
            config_file=config_bundle.config_hydrated_path,
        ),
        UnconfiguredNotebook(
            notebook_path=Path("9xx_figures") / "920_plot-draws",
            raw_notebook_ext=".py",
            summary="figures - Plot draws against each other",
            doc="Create a figure showing the different samples",
            configuration=(),
            dependencies=(config_branch.draw_comparison_table,),
            targets=(config_branch.draw_comparison_figure,),
            config_file=config_bundle.config_hydrated_path,
        ),
    ]

    notebook_output_dir = (
        # Hmmm, need to remove duplicate logic from prefix in dodo.py file
        config_bundle.root_dir_output
        / config_bundle.run_id
        / "noteboooks"
        / branch_name
        / branch_config_id
    )
    notebook_output_dir.mkdir(exist_ok=True, parents=True)

    steps = [
        NotebookStep.from_unconfigured_notebook(
            unconfigured=nb,
            root_dir_raw_notebooks=root_dir_raw_notebooks,
            notebook_output_dir=notebook_output_dir,
            branch_config_id=branch_config_id,
        )
        for nb in figures_notebooks
    ]

    return steps

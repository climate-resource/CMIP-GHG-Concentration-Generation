"""
Covariance plotting notebook steps
"""
from __future__ import annotations

from pathlib import Path

# TODO: move into pydoit_nb so it is more general?
from ..pydoit_nb.notebook_step import NotebookStep
from ..pydoit_nb.notebooks import UnconfiguredNotebook


def get_covariance_plotting_notebook_steps(
    config_bundle,
    branch_name: str,
    branch_config_id: str,
    root_dir_raw_notebooks: Path,
):
    config = config_bundle.config_hydrated

    config_covariance = config.covariance

    covariance_plotting_notebooks = [
        UnconfiguredNotebook(
            notebook_path=Path("3xx_covariance-plotting") / "300_covariance-plotting",
            raw_notebook_ext=".py",
            summary="covariance_plotting - Quick plot to check covariance draws",
            doc="Quick plot to compare covariance draws. Complete plots come later",
            configuration=None,
            dependencies=tuple([c.draw_file for c in config_covariance]),
            targets=(),
            config_file=config_bundle.config_hydrated_path,
        )
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
        for nb in covariance_plotting_notebooks
    ]

    return steps

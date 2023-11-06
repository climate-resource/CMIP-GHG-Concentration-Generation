"""
Covariance notebook steps
"""
from __future__ import annotations

from pathlib import Path

# TODO: move into pydoit_nb so it is more general?
from ..config import get_config_for_branch_id
from ..pydoit_nb.notebook_step import NotebookStep
from ..pydoit_nb.notebooks import UnconfiguredNotebook


def get_covariance_notebook_steps(
    config_bundle,
    branch_name: str,
    branch_config_id: str,
    root_dir_raw_notebooks: Path,
):
    config = config_bundle.config_hydrated

    config_branch = get_config_for_branch_id(
        config=config, branch=branch_name, branch_config_id=branch_config_id
    )
    config_preparation = get_config_for_branch_id(
        config=config, branch="preparation", branch_config_id="only"
    )

    covariance_notebooks = [
        UnconfiguredNotebook(
            notebook_path=Path("1xx_covariance") / "110_draw-samples",
            raw_notebook_ext=".py",
            summary="covariance - draw samples",
            doc="Draw samples with potential covariance",
            configuration=(config_branch.covariance,),
            dependencies=(config_preparation.seed_file,),
            targets=(config_branch.draw_file,),
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
        for nb in covariance_notebooks
    ]

    return steps

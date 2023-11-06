"""
Preparation notebook steps
"""
from __future__ import annotations

from pathlib import Path

# TODO: move into pydoit_nb so it is more general?
from ..config import get_config_for_branch_id
from ..pydoit_nb.notebook_step import NotebookStep
from ..pydoit_nb.notebooks import UnconfiguredNotebook


def get_preparation_notebook_steps(
    config_bundle,
    branch_name: str,
    branch_config_id: str,
    root_dir_raw_notebooks: Path,
):
    config = config_bundle.config_hydrated

    config_branch = get_config_for_branch_id(
        config=config, branch=branch_name, branch_config_id=branch_config_id
    )

    preparation_notebooks = [
        UnconfiguredNotebook(
            notebook_path=Path("0xx_preparation") / "000_write-seed",
            raw_notebook_ext=".py",
            summary="prepare - write seed",
            doc="Write seed for random draws",
            configuration=(config_branch.seed,),
            dependencies=(),
            targets=(config_branch.seed_file,),
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
        for nb in preparation_notebooks
    ]

    return steps

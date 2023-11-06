"""
Task generation related to notebooks
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .config_handling import get_branch_config_ids
from .notebook_step import NotebookStep
from .typing import ConfigBundleLike, Converter, DoitTaskSpec


def get_notebook_branch_tasks(  # noqa: PLR0913
    branch_name: str,
    get_steps: Callable[[ConfigBundleLike, Path], list[NotebookStep]],
    config_bundle: ConfigBundleLike,
    root_dir_raw_notebooks: Path,
    converter: Converter | None = None,
    clean: bool = True,
) -> Iterable[DoitTaskSpec]:
    branch_config_ids = get_branch_config_ids(
        getattr(config_bundle.config_hydrated, branch_name)
    )

    for branch_config_id in branch_config_ids:
        steps = get_steps(
            config_bundle=config_bundle,
            branch_name=branch_name,
            branch_config_id=branch_config_id,
            root_dir_raw_notebooks=root_dir_raw_notebooks,
        )

        for step in steps:
            yield step.to_doit_task(converter=converter, clean=clean)

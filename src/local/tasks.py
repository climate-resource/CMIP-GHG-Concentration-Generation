"""
Task definition and retrieval
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config import ConfigBundle, converter_yaml
from .notebook_steps import get_preparation_notebook_steps
from .pydoit_nb.tasks_notebooks import get_notebook_branch_tasks
from .pydoit_nb.typing import DoitTaskSpec


def gen_all_tasks(
    config_bundle: ConfigBundle,
    root_dir_raw_notebooks: Path,
) -> Iterable[DoitTaskSpec]:
    """
    Generate all tasks in the workflow

    Parameters
    ----------
    config_bundles
        Configuration bundles

    root_dir_raw_notebooks
        Directory in which raw notebooks are kept. The notebook path in the
        elements of `notebook_branch_meta` are assumed to be relative to this
        path.

    Yields
    ------
        :mod:`doit` tasks to run
    """
    tasks = []

    prep_tasks = get_notebook_branch_tasks(
        branch_name="preparation",
        get_steps=get_preparation_notebook_steps,
        config_bundle=config_bundle,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
        converter=converter_yaml,
    )

    tasks.extend(prep_tasks)

    # final_task_targets = []
    # yield from gen_copy_source_into_output_bundle_tasks(
    #     file_dependencies=final_task_targets,
    # )

    yield from tasks

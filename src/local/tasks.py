"""
Task definition and retrieval
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config import converter_yaml
from .config.base import ConfigBundle
from .notebook_steps import (
    analysis,
    constraint,
    covariance,
    covariance_plotting,
    figures,
    grid,
    gridded_data_processing,
    preparation,
    process,
    quick_crunch,
    retrieve,
    write_input4mips,
)
from .pydoit_nb.tasks_copy_source import gen_copy_source_into_output_tasks
from .pydoit_nb.typing import DoitTaskSpec


def gen_all_tasks(
    config_bundle: ConfigBundle,
    root_dir_raw_notebooks: Path,
    repo_root_dir: Path,
    config_file_raw: Path,
) -> Iterable[DoitTaskSpec]:
    """
    Generate all tasks in the workflow

    Parameters
    ----------
    config_bundles
        Configuration bundles

    root_dir_raw_notebooks
        Directory in which raw notebooks are kept. The notebook path in any
        static notebook specifications are assumed to be relative to this path.

    repo_root_dir
        Root directory of the repository, used for copying the source into the
        output path so that a complete bundle can be uploaded easily to Zenodo

    config_file_raw
        Path to the raw configuration file

    Yields
    ------
        :mod:`doit` tasks to run
    """
    notebook_tasks: list[DoitTaskSpec] = []
    for step_module in [
        retrieve,
        process,
        quick_crunch,
        grid,
        gridded_data_processing,
        write_input4mips,
        preparation,
        covariance,
        constraint,
        covariance_plotting,
        analysis,
        figures,
    ]:
        for task in step_module.step.gen_notebook_tasks(
            config_bundle=config_bundle,
            root_dir_raw_notebooks=root_dir_raw_notebooks,
            converter=converter_yaml,
        ):
            yield task
            notebook_tasks.append(task)

    yield from gen_copy_source_into_output_tasks(
        all_preceeding_tasks=notebook_tasks,
        repo_root_dir=repo_root_dir,
        root_dir_output_run=config_bundle.root_dir_output_run,
        run_id=config_bundle.run_id,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
        config_file_raw=config_file_raw,
    )

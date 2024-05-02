"""
Task definition and retrieval
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from pydoit_nb.tasks_copy_source import gen_copy_source_into_output_tasks
from pydoit_nb.typing import DoitTaskSpec

from .config import converter_yaml
from .config.base import ConfigBundle
from .notebook_steps import (
    calculate_ch4_monthly_fifteen_degree_pieces,
    crunch_grids,
    plot_input_data_overviews,
    process_noaa_in_situ_data,
    process_noaa_surface_flask_data,
    retrieve_and_extract_agage_data,
    retrieve_and_extract_ale_data,
    retrieve_and_extract_gage_data,
    retrieve_and_extract_misc_data,
    retrieve_and_extract_noaa_data,
    retrieve_and_process_epica_data,
    retrieve_and_process_law_dome_data,
    retrieve_and_process_neem_data,
    retrieve_and_process_scripps_data,
    smooth_law_dome_data,
    write_input4mips,
)


def copy_no_output(in_path: Path, out_path: Path) -> None:
    """
    Copy a file, producing no output

    Required so that pydoit doesn't think the shutil operation failed.
    Defined here as parallel runs fail if we use pydoit's ``swallow_output`` helper.

    Parameters
    ----------
    in_path
        Source

    out_path
        Destination
    """
    shutil.copy2(src=in_path, dst=out_path)


def copy_tree_no_output(in_path: Path, out_path: Path) -> None:
    """
    Copy a file tree, producing no output

    Required so that pydoit doesn't think the shutil operation failed.
    Defined here as parallel runs fail if we use pydoit's ``swallow_output`` helper.

    Parameters
    ----------
    in_path
        Source

    out_path
        Destination
    """
    shutil.copytree(
        src=in_path,
        dst=out_path,
        ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
        dirs_exist_ok=True,
    )


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
        retrieve_and_extract_misc_data,
        retrieve_and_extract_noaa_data,
        process_noaa_surface_flask_data,
        process_noaa_in_situ_data,
        retrieve_and_extract_agage_data,
        retrieve_and_extract_gage_data,
        retrieve_and_extract_ale_data,
        retrieve_and_process_law_dome_data,
        retrieve_and_process_scripps_data,
        retrieve_and_process_epica_data,
        retrieve_and_process_neem_data,
        plot_input_data_overviews,
        smooth_law_dome_data,
        calculate_ch4_monthly_fifteen_degree_pieces,
        crunch_grids,
        write_input4mips,
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
        copy_file=copy_no_output,
        copy_tree=copy_tree_no_output,
    )

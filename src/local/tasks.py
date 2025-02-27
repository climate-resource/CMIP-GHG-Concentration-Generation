"""
Task definition and retrieval
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from pydoit_nb.tasks_copy_source import (
    gen_copy_source_into_output_tasks,
    get_pydoit_nb_run_command_default,
)
from pydoit_nb.typing import DoitTaskSpec

from .config import converter_yaml
from .config.base import ConfigBundle
from .notebook_steps import (
    calculate_c4f10_like_monthly_fifteen_degree_pieces,
    calculate_c8f18_like_monthly_fifteen_degree_pieces,
    calculate_ch4_monthly_fifteen_degree_pieces,
    calculate_co2_monthly_fifteen_degree_pieces,
    calculate_n2o_monthly_fifteen_degree_pieces,
    calculate_sf6_like_monthly_fifteen_degree_pieces,
    compile_historical_emissions,
    crunch_equivalent_species,
    crunch_grids,
    plot_input_data_overviews,
    process_noaa_hats,
    process_noaa_in_situ_data,
    process_noaa_surface_flask_data,
    retrieve_and_extract_agage_data,
    retrieve_and_extract_ale_data,
    retrieve_and_extract_gage_data,
    retrieve_and_extract_misc_data,
    retrieve_and_extract_noaa_data,
    retrieve_and_process_adam_et_al_2024_data,
    retrieve_and_process_droste_et_al_2020_data,
    retrieve_and_process_epica_data,
    retrieve_and_process_ghosh_et_al_2023_data,
    retrieve_and_process_law_dome_data,
    retrieve_and_process_menking_et_al_2025_data,
    retrieve_and_process_neem_data,
    retrieve_and_process_scripps_data,
    retrieve_and_process_trudinger_et_al_2016_data,
    retrieve_and_process_velders_et_al_2022_data,
    retrieve_and_process_western_et_al_2024_data,
    retrieve_and_process_wmo_2022_ozone_assessment_ch7_data,
    smooth_ghosh_et_al_2023_data,
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


def copy_readme_h(  # noqa: PLR0913
    in_path: Path,
    out_path: Path,
    run_id: str,
    config_file_raw: Path,
    raw_run_instruction: str = "pixi run doit run --verbosity=2",
    get_pydoit_nb_run_command: None = None,
) -> None:
    """
    Copy README into the output bundle

    Parameters
    ----------
    in_path
        Path to the raw README file (normally in the repository's root
        directory)

    out_path
        Path in which to write the README file (normally in the output bundle)

    run_id
        ID of the run. This is injected into the written README as part of the
        footer.

    config_file_raw
        Path to the raw configuration file, relative to the root output
        directory

    raw_run_instruction
        Instructions for how to run the workflow as they appear in the README.
        These are included to check that the instructions for running in the
        bundle are (likely) correct.

    get_pydoit_nb_run_command
        Should always be `None`, we always use the default here

    Raises
    ------
    AssertionError
        The run instructions in the README aren't as expected hence the code
        injection likely won't work as expected.

    ValueError
        `config_file_raw` is an absolute path. It should always be a relative
        path to ensure portability.
    """
    if get_pydoit_nb_run_command is not None:
        raise ValueError(get_pydoit_nb_run_command)

    get_pydoit_nb_run_command_use = get_pydoit_nb_run_command_default

    if config_file_raw.is_absolute():
        msg = f"`config_file_raw` must be a relative path, received: {config_file_raw}"
        raise ValueError(msg)

    with open(in_path) as fh:
        raw = fh.read()

    if raw_run_instruction not in raw:
        msg = (
            "Could not find expected run instructions in README. "
            "The injected run instructions probably won't be correct. "
            f"Expected run instruction: {raw_run_instruction}"
        )
        raise AssertionError(msg)

    footer = f"""
## Pydoit info

This README was created from the raw {in_path.name} file as part of the {run_id!r} run with
[pydoit](https://pydoit.org/contents.html). The bundle should contain
everything required to reproduce the outputs. The environment can be
made with [pixi](https://pixi.sh/latest/) using the `pixi.lock` file
and the `pyproject.toml` file. Please disregard messages about the `Makefile`
in this file.

If you are looking to re-run the analysis, then you should run the below

```sh
{get_pydoit_nb_run_command_use(config_file_raw, raw_run_instruction)}
```

The reason for this is that you want to use the configuration with relative
paths. The configuration file that is included in this output bundle contains
absolute paths for reproducibility and traceability reasons. However, such
absolute paths are not portable so you need to use the configuration file
above, which is the raw configuration file as it was was used in the original
run.

If you have any issues running the analysis, please make an issue in our code
repository or reach out via email.
"""
    with open(out_path, "w") as fh:
        fh.write(raw)
        fh.write(footer)


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
        process_noaa_hats,
        retrieve_and_extract_agage_data,
        retrieve_and_extract_gage_data,
        retrieve_and_extract_ale_data,
        retrieve_and_process_law_dome_data,
        retrieve_and_process_scripps_data,
        retrieve_and_process_epica_data,
        retrieve_and_process_neem_data,
        retrieve_and_process_velders_et_al_2022_data,
        retrieve_and_process_western_et_al_2024_data,
        retrieve_and_process_wmo_2022_ozone_assessment_ch7_data,
        retrieve_and_process_droste_et_al_2020_data,
        retrieve_and_process_adam_et_al_2024_data,
        retrieve_and_process_ghosh_et_al_2023_data,
        retrieve_and_process_menking_et_al_2025_data,
        retrieve_and_process_trudinger_et_al_2016_data,
        plot_input_data_overviews,
        compile_historical_emissions,
        smooth_law_dome_data,
        smooth_ghosh_et_al_2023_data,
        calculate_co2_monthly_fifteen_degree_pieces,
        calculate_ch4_monthly_fifteen_degree_pieces,
        calculate_n2o_monthly_fifteen_degree_pieces,
        calculate_sf6_like_monthly_fifteen_degree_pieces,
        calculate_c4f10_like_monthly_fifteen_degree_pieces,
        calculate_c8f18_like_monthly_fifteen_degree_pieces,
        crunch_grids,
        crunch_equivalent_species,
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
        copy_readme=copy_readme_h,
        other_files_to_copy=(
            "dodo.py",
            "pixi.lock",
            "pyproject.toml",
        ),
    )

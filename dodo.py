"""
[doit](TODO link) configuration file
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from doit import task_params

from local import get_key_info
from local.config import converter_yaml, load_config_from_file
from local.config.base import ConfigBundle
from local.pydoit_nb.config_handling import insert_path_prefix
from local.pydoit_nb.display import print_config
from local.pydoit_nb.serialization import write_config_bundle_to_disk
from local.pydoit_nb.task_parameters import notebook_task_params, run_config_task_params
from local.pydoit_nb.typing import DoitTaskSpec
from local.tasks import gen_all_tasks

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

logFormatter = logging.Formatter(
    "%(levelname)s - %(asctime)s %(name)s %(processName)s (%(module)s:%(funcName)s:%(lineno)d):  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

stdoutHandler = logging.StreamHandler()
stdoutHandler.setFormatter(logFormatter)
stdoutHandler.setLevel(logging.WARNING)

fileHandler = logging.FileHandler("dodo.log")
fileHandler.setFormatter(logFormatter)
# TODO: Make this debug?
fileHandler.setLevel(logging.INFO)

root_logger.addHandler(stdoutHandler)
root_logger.addHandler(fileHandler)

logger = logging.getLogger("dodo")


def print_key_info() -> None:
    """
    Print key information
    """
    key_info = get_key_info().split("\n")
    longest_line = max(len(line) for line in key_info)
    top_line = bottom_line = "=" * longest_line

    print("\n".join([top_line, *key_info, bottom_line]))

    time.sleep(1.5)


def task_display_info() -> dict[str, Any]:
    """
    Display key information

    Returns
    -------
        pydoit task
    """
    return {
        "actions": [print_key_info],
    }


@task_params([*run_config_task_params, *notebook_task_params])
def task_generate_workflow_tasks(
    configuration_file: Path,
    run_id: str,
    root_dir_output: Path,
    root_dir_raw_notebooks: Path,
) -> Iterable[DoitTaskSpec]:
    """
    Generate workflow tasks

    Further description etc. here

    Parameters
    ----------
    configuration_file
        Configuration file to use with this run

    run_id
        The ID for this run

    root_dir_output
        Root directory for outputs

    root_dir_raw_notebooks
        Directory in which the raw (i.e. not yet run or input) notebooks live

    Returns
    -------
        Tasks which can be handled by :mod:`pydoit`
    """
    # TODO: somehow make this happen as part of task_params passing
    configuration_file = configuration_file.absolute()
    root_dir_output = root_dir_output.absolute()
    root_dir_raw_notebooks = root_dir_raw_notebooks.absolute()

    # TODO: decide whether to give user more control over this or not
    root_dir_output_run = root_dir_output / run_id
    root_dir_output_run.mkdir(parents=True, exist_ok=True)

    yield {
        "name": "Show configuration",
        "actions": [
            (
                print_config,
                [],
                dict(
                    configuration_file=configuration_file,
                    run_id=run_id,
                    root_dir_output=root_dir_output,
                    root_dir_raw_notebooks=root_dir_raw_notebooks,
                ),
            )
        ],
    }

    # Current logic: put everything in a single configuration file.
    # The logic (however crazy) for generating that configuration file should
    # be kept separate from actually running all the notebooks to simply
    # maintenance.

    # TODO: decide whether to put these steps together in a 'hydration' function
    config = load_config_from_file(configuration_file)
    config = insert_path_prefix(config, prefix=root_dir_output_run)
    config_bundle = ConfigBundle(
        run_id=run_id,
        config_hydrated=config,
        config_hydrated_path=root_dir_output_run / configuration_file.name,
        root_dir_output=root_dir_output,
        root_dir_output_run=root_dir_output_run,
    )

    write_config_bundle_to_disk(config_bundle=config_bundle, converter=converter_yaml)

    yield {
        "basename": "generate_workflow_tasks",
        "name": None,
        "doc": "Generate tasks for the workflow",
    }

    yield from gen_all_tasks(
        config_bundle,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
    )

    logger.info("Finished generating doit tasks")

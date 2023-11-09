"""
[doit](TODO link) configuration file
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from local import get_key_info
from local.config import converter_yaml, load_config_from_file
from local.config.base import ConfigBundle
from local.pydoit_nb.config_handling import insert_path_prefix
from local.pydoit_nb.display import print_config
from local.pydoit_nb.serialization import write_config_bundle_to_disk
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


def task_generate_workflow_tasks() -> Iterable[DoitTaskSpec]:
    """
    Generate workflow tasks

    This task pulls in the following environment variables:

    - ``DOIT_CONFIGURATION_FILE``
        - The file to use to configure this run

    - ``DOIT_RUN_ID``
        - The ID to use for this run

    - ``DOIT_ROOT_DIR_OUTPUT``
        - The root directory in which to write output

    - ``DOIT_ROOT_DIR_RAW_NOTEBOOKS``
        - The root directory in which the raw (i.e. not yet run) notebooks live

    Returns
    -------
        Tasks which can be handled by :mod:`pydoit`
    """
    # TODO: decide whether to split out this pattern to make it slightly more
    # re-useable
    configuration_file = Path(
        os.environ.get("DOIT_CONFIGURATION_FILE", "dev-config.yaml")
    ).absolute()
    run_id = os.environ.get("DOIT_RUN_ID", dt.datetime.now().strftime("%Y%m%d%H%M%S"))
    root_dir_output = Path(
        os.environ.get("DOIT_ROOT_DIR_OUTPUT", "output-bundles")
    ).absolute()
    root_dir_raw_notebooks = Path(
        os.environ.get("DOIT_ROOT_DIR_RAW_NOTEBOOKS", "notebooks")
    ).absolute()

    # TODO: decide whether to give user more control over this or not
    root_dir_output_run = root_dir_output / run_id
    root_dir_output_run.mkdir(parents=True, exist_ok=True)

    # TOOD: refactor out a re-useable gen_show_configuration_task function
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
        repo_root_dir=Path(__file__).parent,
        config_file_raw=configuration_file,
    )

    logger.info("Finished generating doit tasks")

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
from local.config import converter_yaml, get_config_bundles
from local.pydoit_nb.display import print_config_bundle
from local.pydoit_nb.serialization import write_config_bundle_to_disk
from local.pydoit_nb.task_parameters import notebook_task_params, run_config_task_params
from local.pydoit_nb.tasks import gen_show_config_tasks
from local.pydoit_nb.typing import DoitTaskSpec
from local.tasks import gen_all_tasks

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

logFormatter = logging.Formatter(
    "%(levelname)s - %(asctime)s %(name)s %(processName)s (%(module)s:%(funcName)s:%(lineno)d):  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stdoutHandler = logging.StreamHandler()
stdoutHandler.setFormatter(logFormatter)

root_logger.addHandler(stdoutHandler)

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
    Generate task which displays key information

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
    root_dir_output = root_dir_output.absolute()
    root_dir_raw_notebooks = root_dir_raw_notebooks.absolute()

    # You can add whatever logic and craziness you want above here
    # We recommend at least having the root_dir_output, run_id and
    # raw_notebooks_dir
    # options as these make it easy to do different runs, put output where you
    # want and move the notebooks if you want too. You will always want a line
    # like this that generates your config bundles
    config_bundles = get_config_bundles(
        root_dir_output=root_dir_output,
        run_id=run_id,
    )

    if not config_bundles:
        logger.warning("No configuration bundles")
        return

    # Could move this to a function, but seems silly as probably best for users
    # to control directory creation
    [
        cb.config_hydrated_path.parent.mkdir(exist_ok=True, parents=True)
        for cb in config_bundles
    ]
    [
        write_config_bundle_to_disk(config_bundle=cb, converter=converter_yaml)
        for cb in config_bundles
    ]

    yield from gen_show_config_tasks(config_bundles, print_config_bundle)

    yield from gen_all_tasks(
        config_bundles, root_dir_raw_notebooks=root_dir_raw_notebooks
    )

    logger.info("Finished run")
    print("Finished run")

"""
[doit](TODO link) configuration file
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from doit import task_params

from local import get_key_info
from local.config import converter_yaml, get_config_bundles
from local.pydoit_nb.serialization import write_config_bundle_to_disk
from local.pydoit_nb.task_parameters import run_config_task_params
from local.pydoit_nb.typing import DoitTaskSpec

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


@task_params([*run_config_task_params])
def task_generate_workflow_tasks(
    run_id,
    root_dir_output,
) -> Iterable[DoitTaskSpec]:
    """
    Generate workflow tasks

    Further description etc. here

    Parameters
    ----------
    run_id
        The ID for this run

    root_dir_output
        Root directory for outputs

    raw_notebooks_dir
        Directory in which the raw (i.e. not yet run or input) notebooks live

    Returns
    -------
        Tasks which can be handled by :mod:`pydoit`
    """
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

    [
        write_config_bundle_to_disk(config_bundle=cb, converter=converter_yaml)
        for cb in config_bundles
    ]

    # yield from gen_show_config_tasks(config_bundles)

    # tasks, final_task_targets = get_tasks(config_bundles)
    # yield from process_tasks(tasks)

    # yield from gen_copy_source_into_output_bundle_tasks(
    #     file_dependencies=final_task_targets,
    # )

    # Generate various tasks based on your hydrated configuration
    # There is a pattern here related to making it clear when tasks
    # have:
    # - one dependency and one dependent
    # - multiple dependencies and one dependent
    # - one dependency and multiplie dependents
    # - multiple dependencies and multiple dependents
    # - zero dependencies and zero dependents (and all combos with the above)
    #
    # However I don't know what it is yet

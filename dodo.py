"""
[doit](TODO link) configuration file

TODO: think about whether to move more of this out into pydoit-nb or local
"""
from __future__ import annotations

import datetime as dt
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
from local.pydoit_nb.doit_tools import setup_logging
from local.pydoit_nb.serialization import write_config_bundle_to_disk
from local.pydoit_nb.typing import DoitTaskSpec
from local.tasks import gen_all_tasks

RUN_ID: str = os.environ.get("DOIT_RUN_ID", dt.datetime.now().strftime("%Y%m%d%H%M%S"))
"""ID to use with this run"""

DOIT_CONFIG: dict[str, str] = {
    "backend": os.environ.get("DOIT_DB_BACKEND", "dbm"),
    "dep_file": os.environ.get("DOIT_DB_FILE", f".doit_{RUN_ID}.db"),
}
"""
pydoit configuration

See https://pydoit.org/configuration.html#configuration-at-dodo-py
"""

logger = setup_logging()


def print_key_info() -> None:
    """
    Print key information
    """
    key_info = get_key_info().split("\n")
    longest_line = max(len(line) for line in key_info)
    top_line = bottom_line = "=" * longest_line

    print("\n".join([top_line, *key_info, bottom_line]))

    time.sleep(0.2)


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
    # TODO: put this note somewhere
    # Using environment variables is great because it avoids the pain of
    # doit's weird command-line passing rules and order when doing e.g.
    # `doit list`. However, it does sort of break doit's database because
    # doit's database is keyed based on the task name, not the dependencies
    # (using a json database makes this much much easier to see which is why
    # our dev runs use a json backend).
    # To fix this, I think there's a few options:
    # - do a little hack in here so the database file changes as the run id
    #   changes, this would make the database file be separate for each run id
    #   so avoid the current issue of runs with different run IDs using the
    #   same database hence the up to date status of tasks not being calculated
    #   quite correctly
    #   - Note: this is currently implemented
    # - put the the run ID in the task name so they get stored differently in
    #   the database
    # - something else
    configuration_file = Path(
        os.environ.get("DOIT_CONFIGURATION_FILE", "dev-config.yaml")
    ).absolute()
    run_id = RUN_ID
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

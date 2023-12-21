"""
[Doit](https://pydoit.org) configuration file

This currently contains a few things, but isn't crazy busy. More could be
moved into pydoit-nb but we're currently not doing this until we see which
patterns are actually re-usable.

The key runtime config is currently handled with environment variables. Using
environment variables is great because it avoids the pain of doit's weird
command-line passing rules and order when doing e.g. `doit list`. However, it
does sort of break doit's database because doit's database is keyed based on
the task name, not the dependencies (using a json database makes this much much
easier to see which is why our dev runs use a json backend). To avoid this, I
currently make the database depend on the RUN_ID (see the mangling of
DOIT_CONFIG below). As a result, the database file changes as the run id
changes, so the database file is separate for each run id  and the issue of
different runs using the same database and hence clashing is avoided. This does
feel like a bit of a hack though, not sure if there is a better pattern or
whether this is actually best.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Iterable
from distutils.dir_util import copy_tree
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

    # Give terminal or whatever time to flush
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
    configuration_file = Path(
        os.environ.get("DOIT_CONFIGURATION_FILE", "dev-config.yaml")
    ).absolute()
    # Has to be retrieved earlier so we can set DOIT_CONFIG. I don't love this
    # as we have two patterns, retrieve environment variable into global
    # variable and retrieve environment variable within this function. However,
    # I don't know which way is better so haven't made a choice.
    run_id = RUN_ID
    root_dir_output = Path(
        os.environ.get("DOIT_ROOT_DIR_OUTPUT", "output-bundles")
    ).absolute()
    root_dir_raw_notebooks = Path(
        os.environ.get("DOIT_ROOT_DIR_RAW_NOTEBOOKS", "notebooks")
    ).absolute()

    # TODO: consider giving the user more control over this or not
    root_dir_output_run = root_dir_output / run_id
    root_dir_output_run.mkdir(parents=True, exist_ok=True)

    # TODO: make this handling of raw data a separate task
    # TODO: ask Jared and Mika for thoughts. Copying in full raw data every time
    # seems silly, better to make symlinks at start then only copy when making
    # final bundle?
    # (root_dir_output_run / "data").mkdir(exist_ok=True)
    copy_tree(str(Path("data")), str(root_dir_output_run / "data"))

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
    # be kept separate from actually running all the notebooks to simplify
    # maintenance.
    # TODO: consider putting these steps together in a 'hydration' function
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

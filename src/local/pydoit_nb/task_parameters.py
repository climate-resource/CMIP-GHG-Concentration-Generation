"""
Useful Doit [TODO link] task parameters
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# TODO: move all of these to environment variables so that the listing etc.
# will actually work without so many headaches
run_config_task_params: list[dict[str, Any]] = [
    {
        "name": "configuration_file",
        "default": Path("dev-config.yaml"),
        "type": Path,
        "long": "configuration-file",
        "help": "Path to configuration file",
    },
    {
        "name": "root_dir_output",
        "default": Path("output-bundles"),
        "type": Path,
        "long": "output-root-dir",
        "help": "Root directory for outputs",
    },
    {
        "name": "run_id",
        "default": "run-id-not-set",
        "type": str,
        "long": "run-id",
        "help": "id for the outputs",
    },
]
"""
Task parameters to use to support generating config bundles
"""

notebook_task_params: list[dict[str, Any]] = [
    {
        "name": "root_dir_raw_notebooks",
        "default": Path("notebooks"),
        "type": Path,
        "long": "root-dir-raw-notebooks",
        "help": "Root directory for notebooks",
    },
]
"""
Task parameters to use to support handling notebooks
"""

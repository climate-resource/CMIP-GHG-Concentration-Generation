"""
Useful Doit [TODO link] task parameters
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

run_config_task_params: list[dict[str, Any]] = [
    {
        "name": "root_dir_output",
        "default": Path("output-bundles"),
        "type": Path,
        "long": "output-root-dir",
        "help": "Root directory for outputs",
    },
    {
        "name": "run_id",
        #        "default": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        "default": "zn-test",
        "type": str,
        "long": "run-id",
        "help": "id for the outputs",
    },
]
"""
Task parameters to use to support generating config bundles
"""

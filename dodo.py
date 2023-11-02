"""
[doit](TODO link) configuration file
"""
from __future__ import annotations

import logging
import time
from typing import Any

from local import get_key_info

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

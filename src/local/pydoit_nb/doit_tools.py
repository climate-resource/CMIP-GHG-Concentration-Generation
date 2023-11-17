"""
Tools for compatibility with [doit](pydoit.org)
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any


def setup_logging(
    stdout_level: int = logging.WARNING,
    log_file: str | None = "dodo.log",
    file_level: int = logging.DEBUG,
    log_fmt: str = (
        "%(levelname)s - %(asctime)s %(name)s %(processName)s "
        "(%(module)s:%(funcName)s:%(lineno)d):  %(message)s"
    ),
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Set up logging

    This is a conveniance function. It does things like adding handlers to the
    root logger, which won't always be helpful or desired so use with some care.
    It will not cover all possible logging use cases. If you have a more
    complicated use case, you will probably need to write your own set up
    implementation. However, this implementation may be of some use.

    Parameters
    ----------
    stdout_level
        Level to use for logging to stdout

    log_file
        File to write on disk logs to

    file_level
        Level to use for logging to file

    log_fmt
        Format to use for log strings

    datefmt
        Format to use for dates

    Returns
    -------
        dodo logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    logFormatter = logging.Formatter(
        log_fmt,
        datefmt=datefmt,
    )

    stdoutHandler = logging.StreamHandler()
    stdoutHandler.setFormatter(logFormatter)
    stdoutHandler.setLevel(stdout_level)

    root_logger.addHandler(stdoutHandler)

    if log_file is not None:
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(file_level)

        root_logger.addHandler(fileHandler)

    return logging.getLogger("dodo")


def swallow_output(func: Callable[..., Any]) -> Callable[..., None]:
    """
    Decorate function so the output is swallowed

    This is needed to make pydoit recognise the task has run correctly

    Parameters
    ----------
    func
        Function to decorate

    Returns
    -------
        Decorated function
    """

    @functools.wraps(func)
    def out(*args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    return out

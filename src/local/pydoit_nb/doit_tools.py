"""
Tools for compatibility with [doit](pydoit.org)
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


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

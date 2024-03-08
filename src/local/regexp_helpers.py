"""
Helpers for working with regular expressions
"""

from __future__ import annotations

import re


def re_search_and_retrieve_group(regexp: str, to_search: str, group: str) -> str:
    """
    Search for a regexp in a string and retrieve a specific group
    """
    res = re.search(regexp, to_search)
    if res is None:
        raise ValueError(f"{regexp} not found in {to_search}")  # noqa: TRY003

    return res.group(group)

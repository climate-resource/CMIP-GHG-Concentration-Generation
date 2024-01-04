"""
Re-usable tools that can help when creating configuration

These are likely most helpful for workflows that make use of :mod:`pydoit-nb`.
"""
from __future__ import annotations

from attrs import frozen


@frozen
class URLSource:
    """
    Source information for downloading a source from a URL
    """

    url: str
    """URL to download from"""

    known_hash: str
    """Known hash for the downloaded file"""

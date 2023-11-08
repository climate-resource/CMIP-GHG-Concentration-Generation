"""
Notebook support

TODO: move this to a better name, `notebooks` doesn't make sense given the contents
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

if TYPE_CHECKING:
    from .typing import HandleableConfiguration


@frozen
class UnconfiguredNotebook:
    """A notebook without any configuration"""

    notebook_path: Path
    """Path to notebook, relative to the raw notebook directory"""

    raw_notebook_ext: str
    """Extension for the raw notebook"""

    summary: str
    """One line summary of the notebook"""
    # TODO: validation?

    doc: str
    """Documentation of the notebook (can be longer than one line)"""


@frozen
class ConfiguredNotebook:
    notebook_path: Path
    """Path to notebook, relative to the raw notebook directory"""

    raw_notebook_ext: str
    """Extension for the raw notebook"""

    summary: str
    """One line summary of the notebook"""
    # TODO: validation?

    doc: str
    """Documentation of the notebook (can be longer than one line)"""

    configuration: tuple[HandleableConfiguration, ...] | None
    """
    Configuration used by the notebook.

    If any of the configuration changes then the notebook will be triggered.

    If nothing is provided, then the notebook will be run whenever the
    configuration file driving the notebook is modified (i.e. the notebook will
    be re-run for any configuration change).
    """
    # TODO: It looks like this solves a problem that even the original authors
    # hadn't thought about because they just suggest using forget here
    # https://pydoit.org/cmd-other.html#forget (although they also talk about
    # non-file dependencies elsewhere so maybe these are just out of date docs)

    dependencies: tuple[Path, ...]
    """Paths on which the notebook depends"""

    targets: tuple[Path, ...]
    """Paths which the notebook creates/controls"""

    config_file: Path
    """Path to the config file to use with the notebook"""

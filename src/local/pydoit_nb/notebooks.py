"""
Notebook support
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from attrs import define


@define
class NotebookMetadata:
    """
    Metadata describing a single notebook

    Can later be combined with a configuration bundle to get a :obj:`NotebookStep`
    """

    notebook: Path
    """Path to notebook, relative to the root directory in which the notebooks live"""

    raw_notebook_ext: str
    """Extention that is used with the raw notebook"""

    summary: str
    """
    Short summary of this notebook's functionality
    """

    doc: str
    """Longer description of the notebook"""


@define
class NotebookBranchMetadata:
    """
    Metadata for a collection of notebooks in a branch of the workflow

    [TODO define concept of a branch somewhere, basically just a group of
     related notebooks]
    """

    notebooks: Iterable[NotebookMetadata]

    def to_notebook_meta_dict(self) -> dict[Path, NotebookMetadata]:
        """
        Convert to a dictionary

        Returns
        -------
            Dictionary where each key is the path to the notebook and the
            values are :obj:`NotebookMetadata`
        """
        return {n.notebook: n for n in self.notebooks}

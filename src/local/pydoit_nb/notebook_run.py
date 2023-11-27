"""
Notebook running
"""
from __future__ import annotations

import logging
from pathlib import Path

import jupytext
import papermill

logger = logging.getLogger(__name__)


class NotebookExecutionException(Exception):
    """
    Raised when a notebook fails to execute for any reason
    """

    def __init__(self, exc: Exception, filename: Path):
        note = f"{filename} failed to execute. Original exception: {exc}"
        self.add_note(note)
        super().__init__(exc)


def run_notebook(
    base_notebook: Path,
    unexecuted_notebook: Path,
    executed_notebook: Path,
    notebook_parameters: dict[str, str] | None = None,
) -> None:
    """
    Run a notebook

    This loads the notebook ``base_notebook`` using jupytext, then writes it
    as an ``.ipynb`` file to ``unexecuted_notebook``. It then runs this
    unexecuted notebook with papermill, writing it to ``executed_notebook``.

    Parameters
    ----------
    base_notebook
        Notebook from which to start

    unexecuted_notebook
        Where to write the unexecuted notebook

    executed_notebook
        Where to write the executed notebook

    notebook_parameters
        Parameters to pass to the target notebook

        These parameters will replace the contents of a cell tagged "parameters".
        See the
        `papermill documentation <https://papermill.readthedocs.io/en/latest/usage-parameterize.html#designate-parameters-for-a-cell>`_
        for more information about parameterizing a notebook.

    """
    logger.info("Reading raw notebook with jupytext: %s", base_notebook)
    notebook_jupytext = jupytext.read(base_notebook)

    if notebook_parameters is None:
        notebook_parameters = {}

    logger.info("Writing unexecuted notebook: %s", unexecuted_notebook)
    # TODO: consider whether this should be elsewhere
    unexecuted_notebook.parent.mkdir(parents=True, exist_ok=True)
    jupytext.write(
        notebook_jupytext,
        unexecuted_notebook,
        fmt="ipynb",
    )

    try:
        logger.info("Executing notebook: %s", unexecuted_notebook)
        # TODO: consider whether this should be elsewhere
        executed_notebook.parent.mkdir(parents=True, exist_ok=True)
        papermill.execute_notebook(
            unexecuted_notebook,
            executed_notebook,
            parameters=notebook_parameters,
        )

    except Exception as exc:
        raise NotebookExecutionException(exc, unexecuted_notebook) from exc

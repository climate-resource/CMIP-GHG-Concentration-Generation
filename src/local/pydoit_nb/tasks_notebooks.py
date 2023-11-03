"""
Task generation related to notebooks
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .notebook_step import NotebookStep
from .notebooks import NotebookBranchMetadata
from .typing import ConfigBundleLike, Converter, DoitTaskSpec


def get_notebook_tasks(  # noqa: PLR0913
    notebook_branch_meta: NotebookBranchMetadata,
    config_bundles: Iterable[ConfigBundleLike],
    root_dir_raw_notebooks: Path,
    get_steps: Callable[
        [NotebookBranchMetadata, Iterable[ConfigBundleLike]], list[NotebookStep]
    ],
    common_across_config_bundles: bool = False,
    all_combos_across_config_bundles: bool = False,
    converter: Converter | None = None,
    clean: bool = True,
) -> Iterable[DoitTaskSpec]:
    """
    Get notebook tasks

    This is the key function for generating tasks based on notebooks. It is
    quite powerful, but relies on the user to know how their notebooks should
    be run to get the most out of it.

    Parameters
    ----------
    notebook_branch_meta
        Metadata for the notebooks in the branch

    config_bundles
        Configuration bundles to combine with the notebooks in the branch

    root_dir_raw_notebooks
        Directory in which raw notebooks are kept. The notebook path in the
        elements of `notebook_branch_meta` are assumed to be relative to this
        path.

    get_steps
        Function which combines the notebook metadata and configuration to
        create the :mod:`doit` steps.

    common_across_config_bundles
        Do we expect the steps created by ``get_steps`` to be common across
        each of the different configuration bundles? If yes, set this to
        ``True`` and the function will make sure that this branch is indeed
        the same across all the different configuration bundles.

    all_combos_across_config_bundles
        Do we expect the steps created by ``get_steps`` to be unique?
        If yes, set this to ``True`` and the function will verify that each
        step is unique.

    converter
        Object that can serialise configuration for :mod:`doit` if needed.

    clean
        If we run `doit clean`, should we remove the target of each step?

    Returns
    -------
        :mod:`doit` tasks to run based on each notebook - configuration bundle
        combination (i.e. each step).

    Raises
    ------
    AssertionError
        TODO: fix these errors and descriptions
    """
    if common_across_config_bundles and all_combos_across_config_bundles:
        # TODO: better error
        raise AssertionError

    steps = get_steps(
        notebook_branch_meta=notebook_branch_meta,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
        config_bundles=config_bundles,
    )

    if common_across_config_bundles:
        if len(steps) != 1:
            # TODO: better error message
            raise AssertionError(steps)

    if all_combos_across_config_bundles:
        if len(steps) != len(notebook_branch_meta.notebooks) * len(config_bundles):
            # TODO: better error message
            raise AssertionError(steps)

    for step in steps:
        yield step.to_doit_task(converter=converter, clean=clean)

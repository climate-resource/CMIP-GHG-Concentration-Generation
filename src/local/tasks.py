"""
Task definition and retrieval
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path

from .config import converter_yaml
from .config.base import ConfigBundle
from .notebook_steps.constraint import (
    configure_notebooks_constraint,
    get_unconfigured_notebooks_constraint,
)
from .notebook_steps.covariance import (
    configure_notebooks_covariance,
    get_unconfigured_notebooks_covariance,
)
from .notebook_steps.covariance_plotting import (
    configure_notebooks_covariance_plotting,
    get_unconfigured_notebooks_covariance_plotting,
)
from .notebook_steps.figures import (
    configure_notebooks_figures,
    get_unconfigured_notebooks_figures,
)
from .notebook_steps.preparation import (
    configure_notebooks_prep,
    get_unconfigured_notebooks_prep,
)
from .pydoit_nb.tasks_copy_source import gen_copy_source_into_output_tasks
from .pydoit_nb.tasks_notebooks import get_notebook_branch_tasks
from .pydoit_nb.typing import DoitTaskSpec


def gen_all_tasks(
    config_bundle: ConfigBundle,
    root_dir_raw_notebooks: Path,
    repo_root_dir: Path,
) -> Iterable[DoitTaskSpec]:
    """
    Generate all tasks in the workflow

    Parameters
    ----------
    config_bundles
        Configuration bundles

    root_dir_raw_notebooks
        Directory in which raw notebooks are kept. The notebook path in the
        elements of `notebook_branch_meta` are assumed to be relative to this
        path.

    repo_root_dir
        Root directory of the repository, used for copying the source into the
        output path so that a complete bundle can be uploaded easily to Zenodo

    Yields
    ------
        :mod:`doit` tasks to run
    """
    notebook_tasks: list[DoitTaskSpec] = []
    gnb_tasks = partial(
        get_notebook_branch_tasks,
        config_bundle=config_bundle,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
        converter=converter_yaml,
    )

    prep_tasks = gnb_tasks(
        branch_name="preparation",
        get_unconfigured_notebooks=get_unconfigured_notebooks_prep,
        configure_notebooks=configure_notebooks_prep,
    )
    notebook_tasks.extend(prep_tasks)

    covariance_tasks = gnb_tasks(
        branch_name="covariance",
        get_unconfigured_notebooks=get_unconfigured_notebooks_covariance,
        configure_notebooks=configure_notebooks_covariance,
    )
    notebook_tasks.extend(covariance_tasks)

    constraint_tasks = gnb_tasks(
        branch_name="constraint",
        get_unconfigured_notebooks=get_unconfigured_notebooks_constraint,
        configure_notebooks=configure_notebooks_constraint,
    )
    notebook_tasks.extend(constraint_tasks)

    covariance_plotting_tasks = gnb_tasks(
        branch_name="covariance_plotting",
        get_unconfigured_notebooks=get_unconfigured_notebooks_covariance_plotting,
        configure_notebooks=configure_notebooks_covariance_plotting,
    )
    notebook_tasks.extend(covariance_plotting_tasks)

    figures_tasks = gnb_tasks(
        branch_name="figures",
        get_unconfigured_notebooks=get_unconfigured_notebooks_figures,
        configure_notebooks=configure_notebooks_figures,
    )
    notebook_tasks.extend(figures_tasks)

    yield from notebook_tasks

    yield from gen_copy_source_into_output_tasks(
        all_preceeding_tasks=notebook_tasks,
        repo_root_dir=repo_root_dir,
        root_dir_output_run=config_bundle.root_dir_output_run,
        run_id=config_bundle.run_id,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
    )

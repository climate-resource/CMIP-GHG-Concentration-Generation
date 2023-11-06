"""
Task definition and retrieval
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path

from .config import ConfigBundle, converter_yaml
from .notebook_steps import get_preparation_notebook_steps
from .pydoit_nb.config_handling import get_value_across_config_bundles
from .pydoit_nb.notebook_step import NotebookStep
from .pydoit_nb.tasks_notebooks import get_notebook_branch_tasks
from .pydoit_nb.typing import DoitTaskSpec


def get_prep_notebook_steps(
    notebook_branch_meta: NotebookBranchMetadata,
    root_dir_raw_notebooks: Path,
    config_bundles: Iterable[ConfigBundle],
) -> list[NotebookStep]:
    """
    Get preparation notebook steps

    Parameters
    ----------
    notebook_branch_meta
        Metadata of the notebooks in the preparation branch

    root_dir_raw_notebooks
        Directory in which raw notebooks are kept. The notebook path in the
        elements of `notebook_branch_meta` are assumed to be relative to this
        path.

    config_bundles
        Configuration bundles to run with

    Returns
    -------
        :mod:`doit` tasks related to the preparation branch

    Raises
    ------
    AssertionError
        The steps aren't the same across each configuration bundle (we
        expect the same preparation steps to be required for each
        configuration)
    """
    notebook_meta_dict = notebook_branch_meta.to_notebook_meta_dict()

    gv = partial(
        get_value_across_config_bundles,
        config_bundles=config_bundles,
        expect_all_same=True,
    )
    root_dir_output = gv(access_func=lambda c: c.root_dir_output)
    run_id = gv(access_func=lambda c: c.run_id)
    output_notebook_dir = root_dir_output / run_id

    branch_id = "00x_preparation"
    branch_notebook_dir = output_notebook_dir / branch_id / "notebooks"
    branch_notebook_dir.mkdir(parents=True, exist_ok=True)
    # If assumptions are correct, we can use any config file and will get same
    # result
    branch_config_file = config_bundles[0].config_hydrated_path

    steps = [
        NotebookStep.from_metadata(
            notebook_meta=notebook_meta,
            root_dir_raw_notebooks=root_dir_raw_notebooks,
            # This is the kind of thing we would make clearer in a wrapper
            # function a layer higher than this. Basically, you get rid
            # of the notebook_output_dir and config_id because you should get
            # the same step from all the notebooks. Hence you overwrite these.
            notebook_output_dir=branch_notebook_dir,
            config_id=branch_id,
            configuration=configuration,
            dependencies=dependencies,
            targets=targets,
            config_file=branch_config_file,
        )
        for cb in config_bundles
        for notebook_meta, configuration, dependencies, targets in [
            (
                notebook_meta_dict[Path("0xx_preparation") / "000_write-seed"],
                (cb.config_hydrated.seed,),
                (),
                (cb.config_hydrated.seed_file,),
            )
        ]
    ]

    steps_unique = set(steps)
    if len(steps_unique) != 1:
        # TODO: better error
        raise AssertionError(steps_unique)

    # all steps are same so can simply return first one with certainty
    return steps[:1]


def gen_all_tasks(
    config_bundle: ConfigBundle,
    root_dir_raw_notebooks: Path,
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

    Yields
    ------
        :mod:`doit` tasks to run
    """
    tasks = []

    prep_tasks = get_notebook_branch_tasks(
        branch_name="preparation",
        get_steps=get_preparation_notebook_steps,
        config_bundle=config_bundle,
        root_dir_raw_notebooks=root_dir_raw_notebooks,
        converter=converter_yaml,
    )

    tasks.extend(prep_tasks)

    # final_task_targets = []
    # yield from gen_copy_source_into_output_bundle_tasks(
    #     file_dependencies=final_task_targets,
    # )

    yield from tasks

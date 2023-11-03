"""
Task definition and retrieval
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config import ConfigBundle, converter_yaml
from .pydoit_nb.config_handling import get_value_across_config_bundles
from .pydoit_nb.notebook_step import NotebookStep
from .pydoit_nb.notebooks import NotebookBranchMetadata, NotebookMetadata
from .pydoit_nb.tasks_notebooks import get_notebook_tasks
from .pydoit_nb.typing import DoitTaskSpec

# TODO: move this into separate file
NBM_PREP_STEPS = NotebookBranchMetadata(
    [
        NotebookMetadata(
            notebook=Path("0xx_preparation") / "000_write-seed",
            raw_notebook_ext=".py",
            summary="prepare - write seed",
            doc="Write seed for random draws",
        )
    ]
)


def get_prep_notebook_steps(
    notebook_branch_meta: NotebookBranchMetadata,
    config_bundles: Iterable[ConfigBundle],
) -> list[NotebookStep]:
    """
    Get preparation notebook steps

    Parameters
    ----------
    notebook_branch_meta
        Metadata of the notebooks in the preparation branch

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

    output_notebook_dir = get_value_across_config_bundles(
        config_bundles=config_bundles,
        access_func=lambda c: c.root_dir_output,
        expect_all_same=True,
    )

    branch_id = "00x_preparation"
    branch_notebook_dir = output_notebook_dir / branch_id / "notebooks"
    branch_notebook_dir.mkdir(parents=True, exist_ok=True)
    # If assumptions are correct, we can use any config file and will get same
    # result
    branch_config_file = config_bundles[0].config_hydrated_path

    steps = [
        NotebookStep.from_metadata(
            notebook_meta=notebook_meta,
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
    config_bundles: Iterable[ConfigBundle],
) -> Iterable[DoitTaskSpec]:
    """
    Generate all tasks in the workflow

    Parameters
    ----------
    config_bundles
        Configuration bundles

    Yields
    ------
        :mod:`doit` tasks to run
    """
    tasks = []

    prep_tasks = get_notebook_tasks(
        notebook_branch_meta=NBM_PREP_STEPS,
        config_bundles=config_bundles,
        get_steps=get_prep_notebook_steps,
        common_across_config_bundles=True,
        all_combos_across_config_bundles=False,
        converter=converter_yaml,
    )
    tasks.extend(prep_tasks)

    # final_task_targets = []
    # yield from gen_copy_source_into_output_bundle_tasks(
    #     file_dependencies=final_task_targets,
    # )

    yield from tasks

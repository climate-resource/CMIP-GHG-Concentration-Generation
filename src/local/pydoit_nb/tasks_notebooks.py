"""
Task generation related to notebooks
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeVar

from .config_handling import get_branch_config_ids
from .notebook_step import NotebookStep
from .notebooks import ConfiguredNotebook, UnconfiguredNotebook
from .typing import ConfigBundleLike, Converter, DoitTaskSpec, HandleableConfiguration

T = TypeVar("T")


class GetUnconfiguredNotebooksCallable(Protocol):
    def __call__(self) -> list[UnconfiguredNotebook]:
        ...


class ConfigureNotebooksCallable(Protocol[T]):
    def __call__(
        self,
        unconfigured_notebooks: Iterable[UnconfiguredNotebook],
        config_bundle: ConfigBundleLike[T],
        branch_name: str,
        branch_config_id: str,
    ) -> list[ConfiguredNotebook]:
        ...


def get_notebook_branch_tasks(  # noqa: PLR0913
    branch_name: str,
    get_unconfigured_notebooks: GetUnconfiguredNotebooksCallable,
    configure_notebooks: ConfigureNotebooksCallable[T],
    config_bundle: ConfigBundleLike[T],
    root_dir_raw_notebooks: Path,
    converter: Converter[tuple[HandleableConfiguration, ...]] | None = None,
    clean: bool = True,
) -> Iterable[DoitTaskSpec]:
    unconfigured_notebooks = get_unconfigured_notebooks()
    unconfigured_notebooks_base_tasks = {}
    for nb in unconfigured_notebooks:
        base_task = {
            "basename": f"({nb.notebook_path}) {nb.summary}",
            "name": None,
            "doc": nb.doc,
        }
        yield base_task
        unconfigured_notebooks_base_tasks[nb.notebook_path] = base_task

    branch_config_ids = get_branch_config_ids(
        getattr(config_bundle.config_hydrated, branch_name)
    )

    notebook_output_dir_branch = (
        config_bundle.root_dir_output_run / "notebooks" / branch_name
    )
    for branch_config_id in branch_config_ids:
        configured_notebooks = configure_notebooks(
            unconfigured_notebooks,
            config_bundle=config_bundle,
            branch_name=branch_name,
            branch_config_id=branch_config_id,
        )

        notebook_output_dir_branch_id = notebook_output_dir_branch / branch_config_id
        notebook_output_dir_branch_id.mkdir(exist_ok=True, parents=True)

        for nb_configured in configured_notebooks:
            notebook_task = NotebookStep.from_configured_notebook(
                configured=nb_configured,
                root_dir_raw_notebooks=root_dir_raw_notebooks,
                notebook_output_dir=notebook_output_dir_branch_id,
                branch_config_id=branch_config_id,
            ).to_doit_task(
                base_task=unconfigured_notebooks_base_tasks[nb.notebook_path],
                converter=converter,
                clean=clean,
            )

            yield notebook_task

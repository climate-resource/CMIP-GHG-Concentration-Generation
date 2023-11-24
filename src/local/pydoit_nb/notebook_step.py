"""
Notebook-based step

A notebook-based step is the combination of a notebook and the configuration
to run it.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from attrs import frozen

from local.pydoit_nb.config_handling import get_step_config_ids

from .typing import ConfigBundleLike, Converter, DoitTaskSpec, HandleableConfiguration

if TYPE_CHECKING:
    from .notebook import ConfiguredNotebook, UnconfiguredNotebook


C = TypeVar("C")
CB = TypeVar("CB", contravariant=True)


class ConfigureNotebooksCallable(Protocol[CB]):
    """Callable that can be used for configuring notebooks"""

    def __call__(  # noqa: D102
        self,
        unconfigured_notebooks: Iterable[UnconfiguredNotebook],
        config_bundle: CB,
        step_name: str,
        step_config_id: str,
    ) -> list[ConfiguredNotebook]:
        ...  # pragma: no cover


@frozen
class UnconfiguredNotebookBasedStep(Generic[C]):
    """
    An unconfigured notebook-based step

    A step is a step in the overall workflow. A notebook-based step can be made
    up of one or more notebooks. These are then configured at run-time with the
    run-time information so they can then be turned into doit task(s).
    """

    step_name: str
    """Name of the step"""

    unconfigured_notebooks: list[UnconfiguredNotebook]
    """Unconfigured notebooks that make up this step"""

    configure_notebooks: ConfigureNotebooksCallable[ConfigBundleLike[C]]
    """Function which can configure the notebooks based on run-time information"""

    def gen_notebook_tasks(
        self,
        config_bundle: ConfigBundleLike[C],
        root_dir_raw_notebooks: Path,
        converter: Converter[tuple[HandleableConfiguration, ...]] | None = None,
        clean: bool = True,
    ) -> Iterable[DoitTaskSpec]:
        """
        Generate notebook tasks for this step

        Parameters
        ----------
        config_bundle
            Configuration bundle to use when generating the tasks

        root_dir_raw_notebooks
            Root directory in which the raw notebooks live

        converter
            Instance that can serialise the configuration used by each notebook

        clean
            If we run `doit clean`, should the targets of each task be
            removed?

        Yields
        ------
            Task specifications for use with :mod:`doit`
        """
        unconfigured_notebooks = self.unconfigured_notebooks

        unconfigured_notebooks_base_tasks = {}
        for nb in unconfigured_notebooks:
            base_task = {
                "basename": f"({nb.notebook_path}) {nb.summary}",
                "name": None,
                "doc": nb.doc,
            }
            yield base_task

            unconfigured_notebooks_base_tasks[nb.notebook_path] = base_task

        step_config_ids = get_step_config_ids(
            getattr(config_bundle.config_hydrated, self.step_name)
        )

        notebook_output_dir_step = (
            config_bundle.root_dir_output_run / "notebooks-executed" / self.step_name
        )
        for step_config_id in step_config_ids:
            configured_notebooks = self.configure_notebooks(
                unconfigured_notebooks,
                config_bundle=config_bundle,
                step_name=self.step_name,
                step_config_id=step_config_id,
            )

            notebook_output_dir_step_id = notebook_output_dir_step / step_config_id
            notebook_output_dir_step_id.mkdir(exist_ok=True, parents=True)

            for nb_configured in configured_notebooks:
                notebook_task = nb_configured.to_doit_task(
                    root_dir_raw_notebooks=root_dir_raw_notebooks,
                    notebook_output_dir=notebook_output_dir_step_id,
                    base_task=unconfigured_notebooks_base_tasks[
                        nb_configured.unconfigured_notebook.notebook_path
                    ],
                    converter=converter,
                    clean=clean,
                )

                yield notebook_task

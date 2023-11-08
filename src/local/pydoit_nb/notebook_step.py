"""
Notebook-based step

A notebook-based step is the combination of a notebook and the configuration
to run it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen
from doit.tools import config_changed  # type: ignore

from .notebook_run import run_notebook
from .typing import Converter, DoitTaskSpec, HandleableConfiguration

if TYPE_CHECKING:
    from .notebook import ConfiguredNotebook


@frozen
class NotebookStep:
    """
    A single notebook-configuration step in the workflow

    Each step is the result of combining a notebook with the configuration to
    run it.
    """

    raw_notebook: Path
    """Path to raw notebook"""

    unexecuted_notebook: Path
    """
    Path to unexecuted notebook

    Typically this is inside the output bundle so it's easy to see the
    unexecuted notebook and then compare to the executed version. This can be
    particularly helpful when debugging.
    """

    executed_notebook: Path
    """Path to executed notebook"""

    summary_notebook: str
    """
    Short summary of the notebook
    """

    doc_notebook: str
    """Longer description of the notebook"""

    config_file: Path
    """Path to the config file to use with the notebook"""

    branch_config_id: str
    """`branch_config_id` to use for this run of the notebook"""

    dependencies: tuple[Path, ...]
    """Paths on which the notebook depends"""

    targets: tuple[Path, ...]
    """Paths which the notebook creates/controls"""

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

    @classmethod
    def from_configured_notebook(
        cls,
        configured: ConfiguredNotebook,
        root_dir_raw_notebooks: Path,
        notebook_output_dir: Path,
    ) -> NotebookStep:
        """
        Initialise from a configured notebook

        Parameters
        ----------
        configured
            Configured notebook from which to initialise

        root_dir_raw_notebooks
            Root directory in which the raw (not yet run) notebooks are kept

        notebook_output_dir
            Directory in which to write out the run notebook

        Returns
        -------
            Initialised object
        """
        raw_notebook = root_dir_raw_notebooks / configured.notebook_path.with_suffix(
            configured.raw_notebook_ext
        )
        notebook_name = configured.notebook_path.name

        return cls(
            raw_notebook=raw_notebook,
            unexecuted_notebook=(
                notebook_output_dir / f"{notebook_name}_unexecuted.ipynb"
            ),
            executed_notebook=notebook_output_dir / f"{notebook_name}.ipynb",
            summary_notebook=configured.summary,
            doc_notebook=configured.doc,
            branch_config_id=configured.branch_config_id,
            config_file=configured.config_file,
            dependencies=configured.dependencies,
            targets=configured.targets,
            configuration=configured.configuration,
        )

    def to_doit_task(
        self,
        base_task: DoitTaskSpec,
        converter: Converter[tuple[HandleableConfiguration, ...]] | None = None,
        clean: bool = True,
    ) -> DoitTaskSpec:
        """
        Convert to a :mod:`doit` task

        Parameters
        ----------
        base_task
            Base task definition for this notebook step

        converter
            Converter to use to serialise configuration if needed.

        clean
            If we run `doit clean`, should the targets also be removed?

        Returns
        -------
            Task specification for use with :mod:`doit`

        Raises
        ------
        TypeError
            ``self.configuration is not None`` but ``converter is None``
        """
        dependencies = [
            *self.dependencies,
            self.raw_notebook,
        ]
        notebook_parameters = dict(
            config_file=str(self.config_file), branch_config_id=self.branch_config_id
        )

        targets = self.targets

        task = dict(
            basename=base_task["basename"],
            name=self.branch_config_id,
            doc=f"{base_task['doc']}. branch_config_id={self.branch_config_id!r}",
            actions=[
                (
                    run_notebook,
                    # lambda *args, **kwargs: print(kwargs),
                    [],
                    {
                        "base_notebook": self.raw_notebook,
                        "unexecuted_notebook": self.unexecuted_notebook,
                        "executed_notebook": self.executed_notebook,
                        "notebook_parameters": notebook_parameters,
                    },
                )
            ],
            targets=targets,
            file_dep=dependencies,
            clean=clean,
        )

        if self.configuration is None:
            # Run whenever config file changes
            task["file_dep"].extend([self.config_file])
        else:
            if converter is None:
                # TODO: better error
                raise TypeError(converter)

            has_config_changed = config_changed(
                converter.dumps(self.configuration, sort_keys=True)
            )

            task["uptodate"] = (has_config_changed,)

        return task

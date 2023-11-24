"""
Notebook defining classes
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen
from doit.tools import config_changed  # type: ignore

from .notebook_run import run_notebook

if TYPE_CHECKING:
    from .typing import Converter, DoitTaskSpec, HandleableConfiguration


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
    """
    A configured notebook

    It might make sense to refactor this so has an UnconfiguredNotebook
    as one of its attributes rather than duplicatinng things. That would
    probably also make it clearer that a configured notebook is just the
    combination of an unconfigured notebook and the config we want to use
    with it.
    """

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

    step_config_id: str
    """`step_config_id` to use for this run of the notebook"""

    def to_doit_task(  # noqa: PLR0913
        self,
        root_dir_raw_notebooks: Path,
        notebook_output_dir: Path,
        base_task: DoitTaskSpec,
        converter: Converter[tuple[HandleableConfiguration, ...]] | None = None,
        clean: bool = True,
    ) -> DoitTaskSpec:
        """
        Convert to a :mod:`doit` task

        Parameters
        ----------
        root_dir_raw_notebooks
            Root directory in which the raw (not yet run) notebooks are kept

        notebook_output_dir
            Directory in which to write out the run notebook

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
        raw_notebook = root_dir_raw_notebooks / self.notebook_path.with_suffix(
            self.raw_notebook_ext
        )

        notebook_name = self.notebook_path.name
        unexecuted_notebook = notebook_output_dir / f"{notebook_name}_unexecuted.ipynb"
        executed_notebook = notebook_output_dir / f"{notebook_name}.ipynb"

        dependencies = [
            *self.dependencies,
            raw_notebook,
        ]
        notebook_parameters = dict(
            config_file=str(self.config_file), step_config_id=self.step_config_id
        )

        targets = self.targets

        task = dict(
            basename=base_task["basename"],
            name=self.step_config_id,
            doc=f"{base_task['doc']}. step_config_id={self.step_config_id!r}",
            actions=[
                (
                    run_notebook,
                    # lambda *args, **kwargs: print(kwargs),
                    [],
                    {
                        "base_notebook": raw_notebook,
                        "unexecuted_notebook": unexecuted_notebook,
                        "executed_notebook": executed_notebook,
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

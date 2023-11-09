"""
Generation of tasks for copying source into the outputs
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from attrs import frozen

from .doit_tools import swallow_output
from .typing import DoitTaskSpec


@frozen
class ActionDef:
    """Definition of an action"""

    name: str
    """Name of the action"""

    action: tuple[Callable[..., Any], list[Any], dict[str, Any]]
    """Action to execute with doit"""

    targets: tuple[Path, ...]
    """Files that this action creates"""


def gen_copy_source_into_output_tasks(  # noqa: PLR0913
    all_preceeding_tasks: Iterable[DoitTaskSpec],
    repo_root_dir: Path,
    root_dir_output_run: Path,
    run_id: str,
    root_dir_raw_notebooks: Path,
    config_file_raw: Path,
    readme: str = "README.md",
    zenodo: str = "zenodo.json",
    other_files_to_copy: tuple[str, ...] = (
        "dodo.py",
        "poetry.lock",
        "pyproject.toml",
    ),
    src_dir: str = "src",
) -> Iterable[DoitTaskSpec]:
    """
    Generate tasks to copy the source into the output directory

    Parameters
    ----------
    all_preceeding_tasks
        All tasks preceeding this one. The targets of these tasks are set
        as dependencies of this task to ensure that this task runs after them.

    repo_root_dir
        Root directory of the repository. This is used to know where to copy
        files from.

    root_dir_output_run
        Root directory of the run's output.

    run_id
        ID of the run.

    root_dir_raw_notebooks
        Root directory to the raw notebooks (these are copied into the output
        bundle to ensure that the bundle can be run standalone)

    config_file_raw
        Path to the raw configuration file

    config_file_raw_output_name
        Name to use when saving the raw configuration file to the output bundle
        (must be different to the hydrated configuration file to avoid a clash)

    readme
        Name of the README file to copy into the output

    zenodo
        Name of the zenodo JSON file to copy into the output

    other_files_to_copy
        Other files to copy into the output (paths are relative to the
         project's root)

    src_dir
        Path to the Python source (this is also copied into the output bundle)

    Returns
    -------
        Tasks for copying the source files into the output directory
    """
    all_targets = []
    for task in all_preceeding_tasks:
        if "targets" in task:
            all_targets.extend(task["targets"])

    base_task = {
        "basename": "copy_source_into_output",
        "doc": (
            "Copy required source files into the output directory, making it "
            "easy to create a neat bundle for uploading to Zenodo"
        ),
    }

    output_dir_raw_notebooks = root_dir_output_run / root_dir_raw_notebooks.relative_to(
        repo_root_dir
    )

    config_file_raw_output = (
        root_dir_output_run / f"{config_file_raw.stem}-raw{config_file_raw.suffix}"
    )

    action_defs = [
        ActionDef(
            name="copy README",
            action=(
                copy_readme,
                [
                    repo_root_dir / readme,
                    root_dir_output_run / readme,
                    run_id,
                    config_file_raw_output.relative_to(root_dir_output_run),
                ],
                {},
            ),
            targets=(root_dir_output_run / readme,),
        ),
        ActionDef(
            name="copy Zenodo",
            action=(
                copy_zenodo,
                [repo_root_dir / zenodo, root_dir_output_run / zenodo, run_id],
                {},
            ),
            targets=(root_dir_output_run / zenodo,),
        ),
        *[
            ActionDef(
                name=f"copy {file_name}",
                action=(
                    swallow_output(shutil.copy2),
                    [repo_root_dir / file_name, root_dir_output_run / file_name],
                    {},
                ),
                targets=(root_dir_output_run / file_name,),
            )
            for file_name in other_files_to_copy
        ],
        ActionDef(
            name="copy raw config",
            action=(
                swallow_output(shutil.copy2),
                [config_file_raw, config_file_raw_output],
                {},
            ),
            targets=(config_file_raw_output,),
        ),
        ActionDef(
            name="copy raw notebooks",
            action=(
                swallow_output(shutil.copytree),
                [root_dir_raw_notebooks, output_dir_raw_notebooks],
                dict(
                    ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
                    dirs_exist_ok=True,
                ),
            ),
            targets=(output_dir_raw_notebooks,),
        ),
        ActionDef(
            name="copy source",
            action=(
                swallow_output(shutil.copytree),
                [repo_root_dir / src_dir, root_dir_output_run / src_dir],
                dict(
                    ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
                    dirs_exist_ok=True,
                ),
            ),
            targets=(root_dir_output_run / src_dir,),
        ),
    ]

    for action_def in action_defs:
        created_files_short = tuple(f".../{t.name}" for t in action_def.targets)
        yield {
            "basename": base_task["basename"],
            "doc": f"{base_task['doc']}. Copying in {created_files_short}",
            "name": action_def.name,
            "actions": [action_def.action],
            "targets": action_def.targets,
            "file_dep": all_targets,
        }


def copy_readme(
    in_path: Path, out_path: Path, run_id: str, config_file_raw: Path
) -> None:
    """
    Copy the README into the output bundle

    The footer is currently hard-coded, but this could obviously be opened up
    (and perhaps should be sooner rather than later).

    Parameters
    ----------
    in_path
        Path to the raw README file (normally in the repository's root
        directory)

    out_path
        Path in which to write the README file (normally in the output bundle)

    run_id
        ID of the run. This is injected into the written README as part of the
        footer.

    config_file_raw
        Path to the raw configuration file, relative to the root output
        directory

    Raises
    ------
    AssertionError
        The run instructions in the README isn't as expected hence the code
        injection likely won't work as expected.
    """
    with open(in_path) as fh:
        raw = fh.read()

    raw_run_instruction = "poetry run doit run --verbosity=2"
    if raw_run_instruction not in raw:
        raise AssertionError(  # noqa: TRY003
            "Could not find expected run instructions in README. "
            "The injected run instructions probably won't be correct. "
            f"Expected run instruction: {raw_run_instruction}"
        )

    footer = f"""
## Pydoit info

This README was created from the raw {in_path.name} file as part of the {run_id!r} run with
[pydoit](https://pydoit.org/contents.html). The bundle should contain
everything required to reproduce the outputs. The environment can be
made with [poetry](https://python-poetry.org/) using the `poetry.lock` file
and the `pyproject.toml` file. Please disregard messages about the `Makefile`
in this file.

If you are looking to re-run the analysis, then you should run the below

```sh
DOIT_CONFIGURATION_FILE={config_file_raw} {raw_run_instruction}
```

The reason for this is that you want to use the configuration with relative
paths. The configuration file that is included in this output bundle contains
absolute paths for reproducibility and traceability reasons. However, such
absolute paths are not portable so you need to use the configuration file
above, which is the raw configuration file as it was was used in the original
run.
"""
    with open(out_path, "w") as fh:
        fh.write(raw)
        fh.write(footer)


def copy_zenodo(in_path: Path, out_path: Path, version: str) -> None:
    """
    Copy Zenodo JSON file to the output bundle

    This updates the version information too

    TODO: link to Zenodo JSON docs on available fields in zenodo.json so it is
          easier to update for users like us

    Parameters
    ----------
    in_path
        Path to raw Zenodo file

    out_path
        Path to output Zenodo file in the bundle

    version
        Version to write in the Zenodo file
    """
    with open(in_path) as fh:
        zenodo_metadata = json.load(fh)

    zenodo_metadata["metadata"]["version"] = version

    with open(out_path, "w") as fh:
        fh.write(json.dumps(zenodo_metadata, indent=2))

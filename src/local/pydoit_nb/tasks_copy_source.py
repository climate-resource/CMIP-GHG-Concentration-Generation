"""
Generation of tasks for copying source into the outputs
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from attrs import frozen

from .doit_tools import swallow_output


@frozen
class ActionDef:
    """Definition of an action"""

    name: str
    """Name of the action"""

    action: tuple[Callable[[...], Any], list[Any], dict[str, Any]]
    """Action to execute with doit"""

    targets: tuple[Path, ...]
    """Files that this action creates"""


def gen_copy_source_into_output_tasks(
    all_tasks,
    repo_root_dir,
    root_dir_output_run,
    run_id,
    readme: str = "README.md",
    zenodo: str = "zenodo.json",
    other_files_to_copy: tuple[str, ...] = (
        "dodo.py",
        "poetry.lock",
        "pyproject.toml",
    ),
    src_dir: str = "src",
):
    all_targets = []
    for task in all_tasks:
        if "targets" in task:
            all_targets.extend(task["targets"])

    base_task = {
        "basename": "copy_source_into_output",
        "doc": (
            "Copy required source files into the output directory, making it "
            "easy to create a neat bundle for uploading to Zenodo",
        ),
    }

    action_defs = [
        ActionDef(
            name="copy README",
            action=(
                copy_readme,
                [repo_root_dir / readme, root_dir_output_run / readme, run_id],
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


def copy_readme(in_path: Path, out_path: Path, run_id: str) -> None:
    with open(in_path) as fh:
        raw = fh.read()

    footer = f"""
## Pydoit info

This README was created from the raw {in_path.name} file as part of the {run_id!r} run with
[pydoit](https://pydoit.org/contents.html). The bundle should contain
everything required to reproduce the outputs. The environment can be
made with [poetry](https://python-poetry.org/) using the `poetry.lock` file
and the `pyproject.toml` file. Please disregard messages about the `Makefile`
in this file."""
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

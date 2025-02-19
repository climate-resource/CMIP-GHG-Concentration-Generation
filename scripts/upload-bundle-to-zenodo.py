"""
Upload a bundle to Zenodo
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Annotated

import typer


# ruff: noqa: D101, D102, D103
def create_tar_file(
    original_bundle_path: Path,
    current_level_path: Path,
    zenodo_bundle_path: Path,
    exclude_filters: list[str],
    files_only: bool = False,
) -> None:
    tar_id = "--".join(current_level_path.parts)
    tar_path = zenodo_bundle_path / f"{tar_id}.tar.gz"
    print(f"Writing to {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        for file_to_keep_candidate in (original_bundle_path / current_level_path).iterdir():
            if files_only and not file_to_keep_candidate.is_file():
                continue

            if any(ef in str(file_to_keep_candidate) for ef in exclude_filters):
                continue

            print(f"    - Adding {file_to_keep_candidate}")
            tar.add(file_to_keep_candidate)


def create_level_aware_tar(  # noqa: PLR0913
    original_bundle_path: Path,
    current_level_path: Path,
    zenodo_bundle_path: Path,
    current_level: int,
    copy_at_level: int,
    exclude_filters: list[str],
) -> None:
    if copy_at_level == current_level:
        create_tar_file(
            original_bundle_path=original_bundle_path,
            current_level_path=current_level_path,
            zenodo_bundle_path=zenodo_bundle_path,
            exclude_filters=exclude_filters,
        )

        return

    level_files_l = []
    for level_file in (original_bundle_path / current_level_path).iterdir():
        if any(ef in str(level_file) for ef in exclude_filters):
            continue

        level_file_rel = level_file.relative_to(original_bundle_path)
        if level_file.is_dir():
            create_level_aware_tar(
                original_bundle_path=original_bundle_path,
                current_level_path=level_file_rel,
                zenodo_bundle_path=zenodo_bundle_path,
                current_level=current_level + 1,
                copy_at_level=copy_at_level,
                exclude_filters=exclude_filters,
            )

        else:
            level_files_l.append(level_file)

    if level_files_l:
        create_tar_file(
            original_bundle_path=original_bundle_path,
            current_level_path=current_level_path,
            zenodo_bundle_path=zenodo_bundle_path,
            exclude_filters=exclude_filters,
            files_only=True,
        )


def create_zenodo_bundle(zenodo_bundle_path: Path, original_bundle_path: Path) -> None:
    # TODO: remove hard-coding
    file_globs_to_copy = (
        "README.md",
        "dodo.py",
        "pixi.lock",
        "pyproject.toml",
        "*.yaml",
    )

    directories_to_copy = (
        ("src", 0, ["egg"]),
        ("notebooks", 1, ["ipynb_checkpoints"]),
        (
            "data",
            2,
            [
                "complete",
                "checklist",
                "PRIMAP",  # Backed by DOI
                "epica",  # Backed by DOI
                "law_dome",  # Backed by DOI
                "neem",  # Backed by DOI
                "natural_earth",
            ],
        ),
    )

    zenodo_bundle_path.mkdir(exist_ok=True, parents=True)

    for fg in file_globs_to_copy:
        for file in original_bundle_path.glob(fg):
            shutil.copyfile(file, zenodo_bundle_path / file.name)

    for dc, copy_at_level, exclude_filters in directories_to_copy:
        create_level_aware_tar(
            original_bundle_path=original_bundle_path,
            current_level_path=Path(dc),
            zenodo_bundle_path=zenodo_bundle_path,
            current_level=0,
            copy_at_level=copy_at_level,
            exclude_filters=exclude_filters,
        )


def main(
    bundle_path: Annotated[Path, typer.Argument(help="Path to the bundle to upload")],
    zenodo_bundle_root_path: Annotated[
        Path, typer.Option(help="Root path in which to save Zenodo bundles")
    ] = Path("zenodo-bundles"),
) -> None:
    bundle_id = bundle_path.parts[-1]
    zenodo_bundle_path = zenodo_bundle_root_path / bundle_id

    create_zenodo_bundle(zenodo_bundle_path=zenodo_bundle_path, original_bundle_path=bundle_path)


if __name__ == "__main__":
    typer.run(main)

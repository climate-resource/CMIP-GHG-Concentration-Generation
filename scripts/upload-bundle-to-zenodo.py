"""
Upload a bundle to Zenodo
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger
from openscm_zenodo.zenodo import ZenodoInteractor


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
        ("src", 0, ["egg", "DS_Store"]),
        ("notebooks", 1, ["ipynb_checkpoints", "DS_Store"]),
        (
            "data",
            2,
            [
                "DS_Store",
                "complete",
                "checklist",
                "primap",  # Backed by DOI
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


def upload_to_zenodo(
    zenodo_bundle_path: Path,
    publish: bool,
    zenodo_interactor: ZenodoInteractor,
    draft_deposition_id: str,
    metadata: dict[str, Any],
) -> None:
    zenodo_interactor.update_metadata(deposition_id=draft_deposition_id, metadata=metadata)

    zenodo_interactor.remove_all_files(deposition_id=draft_deposition_id)

    bucket_url = zenodo_interactor.get_bucket_url(deposition_id=draft_deposition_id)
    for file in zenodo_bundle_path.iterdir():
        zenodo_interactor.upload_file_to_bucket_url(
            file,
            bucket_url=bucket_url,
        )

    if publish:
        zenodo_interactor.publish(deposition_id=draft_deposition_id)
        print(f"Published the new record at https://zenodo.org/records/{draft_deposition_id}")

    else:
        print(f"You can preview the draft upload at https://zenodo.org/uploads/{draft_deposition_id}")


def main(  # noqa: PLR0913
    bundle_path: Annotated[Path, typer.Argument(help="Path to the bundle to upload")],
    zenodo_bundle_root_path: Annotated[
        Path, typer.Option(help="Root path in which to save Zenodo bundles")
    ] = Path("zenodo-bundles"),
    publish: Annotated[bool, typer.Option(help="Should we publish the uploaded data?")] = False,
    logging_level: Annotated[str, typer.Option(help="Logging level to use")] = "INFO",
    zenodo_metadata_file: Annotated[
        str, typer.Option(help="Name of the file in which the zenodo metadata was written")
    ] = "zenodo.json",
    reserved_zenodo_doi_file: Annotated[
        str, typer.Option(help="Name of the file in which the reserved Zenodo DOI was saved")
    ] = "reserved-zenodo-doi.txt",
) -> None:
    load_dotenv()

    logger.configure(handlers=[dict(sink=sys.stderr, level=logging_level)])
    logger.enable("openscm_zenodo")

    bundle_id = bundle_path.parts[-1]
    zenodo_bundle_path = zenodo_bundle_root_path / bundle_id
    zenodo_interactor = ZenodoInteractor(token=os.environ["ZENODO_TOKEN"])

    with open(bundle_path / zenodo_metadata_file) as fh:
        zenodo_metadata = json.load(fh)

    with open(bundle_path / f"{bundle_id}-config.yaml") as fh:
        config = yaml.safe_load(fh)

    draft_deposition_id = config["doi"].split("10.5281/zenodo.")[1]

    create_zenodo_bundle(zenodo_bundle_path=zenodo_bundle_path, original_bundle_path=bundle_path)

    upload_to_zenodo(
        zenodo_bundle_path,
        publish=publish,
        zenodo_interactor=zenodo_interactor,
        draft_deposition_id=draft_deposition_id,
        metadata=zenodo_metadata,
    )


if __name__ == "__main__":
    typer.run(main)

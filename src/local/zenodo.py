"""
Tools for interacting with Zenodo
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from openscm_zenodo.zenodo import ZenodoDomain, ZenodoInteractor, get_reserved_doi


def get_zenodo_doi(any_deposition_id: str) -> str:
    """
    Get Zenodo DOI

    This does everything in one.
    If you want more control, copy the code and break the function up.

    Parameters
    ----------
    any_deposition_id
        Any deposition ID in the Zenodo series

    Returns
    -------
    :
        Zenodo DOI
    """
    load_dotenv()

    zenoodo_interactor = ZenodoInteractor(
        token=os.environ["ZENODO_TOKEN"],
        zenodo_domain=ZenodoDomain.production.value,
    )

    latest_deposition_id = zenoodo_interactor.get_latest_deposition_id(
        any_deposition_id=any_deposition_id,
    )

    try:
        new_deposition_id = zenoodo_interactor.create_new_version_from_latest(
            latest_deposition_id=latest_deposition_id
        ).json()["id"]
    except AssertionError:
        # Assume that our latest draft is fine
        new_deposition_id = latest_deposition_id

    try:
        zenoodo_interactor.remove_all_files(deposition_id=new_deposition_id)
    except requests.exceptions.HTTPError:
        # assume all files already removed and move on
        pass

    metadata = zenoodo_interactor.get_metadata(latest_deposition_id)
    metadata["metadata"]["prereserve_doi"] = True  # type: ignore

    update_metadata_response = zenoodo_interactor.update_metadata(
        deposition_id=new_deposition_id,
        metadata=metadata,
    )

    reserved_doi = get_reserved_doi(update_metadata_response)

    return reserved_doi

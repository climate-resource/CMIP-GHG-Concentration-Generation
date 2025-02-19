"""
Tools for interacting with Zenodo
"""

from __future__ import annotations

import os

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
    draft_deposition_id = zenoodo_interactor.get_draft_deposition_id(
        latest_deposition_id=latest_deposition_id
    )

    metadata = zenoodo_interactor.get_metadata(latest_deposition_id, user_controlled_only=True)
    for k in ["doi", "prereserve_doi", "publication_date", "version"]:
        if k in metadata["metadata"]:
            metadata["metadata"].pop(k)

    update_metadata_response = zenoodo_interactor.update_metadata(
        deposition_id=draft_deposition_id,
        metadata=metadata,
    )

    doi = get_reserved_doi(update_metadata_response)

    return doi

"""
Create config for the Zenodo deposition creation step
"""

from __future__ import annotations

from pathlib import Path

from local.config.zenodo import CreateZenodoDepositionConfig


def create_create_zenodo_deposition_config(
    any_deposition_id: str = "13365838",
    reserved_zenodo_doi_file: Path = Path("reserved-zenodo-doi.txt"),
) -> list[CreateZenodoDepositionConfig]:
    """
    Create configuration for creating the next Zenodo deposition

    Parameters
    ----------
    any_deposition_id
        Deposition ID for any deposition which has been published in the Zenodo series

    reserved_zenodo_doi_file
        Path in which to save the reserved Zenodo DOI

    Returns
    -------
        Created configuration
    """
    return [
        CreateZenodoDepositionConfig(
            step_config_id="only",
            any_deposition_id=any_deposition_id,
            reserved_zenodo_doi_file=reserved_zenodo_doi_file,
        )
    ]

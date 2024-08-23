"""
Config for the Zenodo deposition creation step
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CreateZenodoDepositionConfig:
    """
    Configuration class for creating the Zenodo deposition
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    any_deposition_id: str
    """Deposition ID for any deposition which has been published in the Zenodo series"""

    reserved_zenodo_doi_file: Path
    """Path in which to save the reserved Zenodo DOI"""

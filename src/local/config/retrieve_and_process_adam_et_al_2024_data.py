"""
Config for retrieving, extracting and processing Adam et al. (2024) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class RetrieveExtractAdamEtal2024Data:
    """
    Config for retrieving, extracting and processing Adam et al. (2024) data

    Original paper: https://doi.org/10.1038/s43247-024-01946-y
    Raw data given to us by Luke Western
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    raw_data_file: Path
    """
    Path to the raw data file
    """

    processed_data_file: Path
    """File in which to save the processed data"""

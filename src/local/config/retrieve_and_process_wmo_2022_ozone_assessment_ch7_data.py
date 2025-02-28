"""
Config for retrieving, extracting and processing WMO 2022 ozone assessment ch. 7 data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen

from local.dependencies import SourceInfo


@frozen
class RetrieveProcessWMO2022OzoneAssessmentCh7Config:
    """
    Config for retrieving, extracting and processing WMO 2022 ozone assessment ch. 7 data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    raw_data: Path
    """
    File in which the raw data is saved

    This file is not available to be downloaded,
    so we include the data in the repository
    for reproducibility purposes.
    """

    processed_data_file: Path
    """File in which to save the processed data"""

    source_info: SourceInfo
    """Source information"""

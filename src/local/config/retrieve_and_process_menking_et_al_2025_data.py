"""
Config for retrieving, extracting and processing Menking et al. (2025) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen

from local.dependencies import SourceInfo


@frozen
class RetrieveExtractMenkingEtal2025Data:
    """
    Config for retrieving, extracting and processing Menking et al. (2025) data

    Included here as the paper was in prep at the time of preparing the data.
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

    expected_hash: str
    """Expected hash of the file we're loading"""

    processed_data_file: Path
    """File in which to save the processed data"""

    source_info: SourceInfo
    """Source information"""

    second_order_deps: dict[str, tuple[SourceInfo, ...]]
    """
    Second-order dependency source information

    Keys are gases, values are second-order dependencies.
    """

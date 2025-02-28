"""
Config for retrieving, extracting and processing Droste et al. (2020) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource

from local.dependencies import SourceInfo


@frozen
class RetrieveExtractDrosteEtal2020Data:
    """
    Config for retrieving, extracting and processing Droste et al. (2020) data

    Original paper: https://doi.org/10.5194/acp-20-4787-2020
    Zenodo record: https://zenodo.org/records/3519317
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    zenodo_record: URLSource
    """Zenodo record from which to download the raw data"""

    raw_dir: Path
    """
    File in which to save the raw data
    """

    download_complete_file: Path
    """
    Path in which to write the time at which the download was completed

    This is mainly used to help with setting the dependencies between notebooks correctly.
    """

    processed_data_file: Path
    """File in which to save the processed data"""

    source_info: SourceInfo = SourceInfo(
        short_name="Droste et al., 2020",
        licence="CC BY 4.0",  # https://zenodo.org/records/3519317
        reference=(
            "Droste, E. S., Adcock, K. E., ..., Sturges, W. T., and Laube, J. C.: "
            "Trends and emissions of six perfluorocarbons "
            "in the Northern Hemisphere and Southern Hemisphere, "
            "Atmos. Chem. Phys., 20, 4787-4807, https://doi.org/10.5194/acp-20-4787-2020, 2020."
        ),
        doi="https://doi.org/10.5194/acp-20-4787-2020",
        url="https://doi.org/10.5194/acp-20-4787-2020",
        resource_type="publication-article",
    )
    """Source information"""

"""
Config for processing NOAA in-situ data
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class ProcessNOAAInSituDataConfig:
    """
    Configuration class for processing NOAA in-situ data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas for which we are processing data"""

    processed_monthly_data_with_loc_file: Path
    """
    Where to save the processed monthly data

    This data should include location i.e. latitude and longitude information
    and have one entry per station per month (i.e. do average over all
    observations for the station in a month before saving).
    """

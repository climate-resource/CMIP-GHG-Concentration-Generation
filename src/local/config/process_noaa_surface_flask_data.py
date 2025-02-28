"""
Config for processing NOAA surface flask data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class ProcessNOAASurfaceFlaskDataConfig:
    """
    Configuration class for processing NOAA surface flask data
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

    source_info_short_names_file: Path
    """Path in which to save the short names of all sources used to compile the dataset"""

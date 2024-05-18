"""
Config for the write input4MIPs file step
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class WriteInput4MIPsConfig:
    """
    Configuration class for the write input4MIPs step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas for which we are writing files"""

    start_year: int
    """
    First year for which to write data

    Used to ensure that all data goes over the same time period
    because some data providers update faster than others.
    Also allows us to ensure that all data covers the same time axis.
    """

    end_year: int
    """
    Last year for which to write data

    Used to ensure that all data goes over the same time period
    because some data providers update faster than others.
    Also allows us to ensure that all data covers the same time axis.
    """

    input4mips_out_dir: Path
    """Path in which to save the processed input4MIPs-ready data"""

    complete_file: Path
    """Path in which to save the timestamp of the time at which this step was completed"""

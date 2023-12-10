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

    input4mips_out_dir: Path
    """Path in which to save the processed input4MIPs-ready data"""

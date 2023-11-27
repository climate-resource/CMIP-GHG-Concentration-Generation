"""
Config for the constraint step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class ConstraintConfig:
    """
    Configuration class for the constraint step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    constraint_gradient: float
    """Gradient to use when creating the constraint"""

    draw_file: Path
    """Path in which to save the drawn data"""

"""
Config for the figures step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class FiguresConfig:
    """
    Configuration class for the figures step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    misc_figures_dir: Path
    """Directory in which to save miscellaneous figures"""

    draw_comparison_table: Path
    """Path in which to save the table with all the draws in one"""

    draw_comparison_figure: Path
    """Path in which to save the figure comparing the draws"""

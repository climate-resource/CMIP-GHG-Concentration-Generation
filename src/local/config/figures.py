"""
Config for the figures branch
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class FiguresConfig:
    """
    Configuration class for the figures branch
    """

    branch_config_id: str
    """
    ID for this configuration of the branch

    Must be unique among all configurations for this branch
    """

    misc_figures_dir: Path
    """Directory in which to save miscellaneous figures"""

    draw_comparison_table: Path
    """Path in which to save the table with all the draws in one"""

    draw_comparison_figure: Path
    """Path in which to save the figure comparing the draws"""

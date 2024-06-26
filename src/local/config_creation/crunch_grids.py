"""
Create config for crunching different grids
"""

from __future__ import annotations

from pathlib import Path

from local.config.crunch_grid import GridCrunchingConfig


def create_crunch_grids_config(gases: tuple[str, ...]) -> list[GridCrunchingConfig]:
    """
    Create configuration for crunching the gridded data for different gases

    Parameters
    ----------
    gases
        Gases for which to create grid crunching configurations

    Returns
    -------
        Grid crunching configurations for the requested gases
    """
    out = []

    for gas in gases:
        interim_dir = Path("data/interim") / gas

        out.append(
            GridCrunchingConfig(
                step_config_id=gas,
                gas=gas,
                fifteen_degree_monthly_file=interim_dir
                / f"{gas}_fifteen-degree_monthly.nc",
                half_degree_monthly_file=interim_dir / f"{gas}_half-degree_monthly.nc",
                gmnhsh_mean_monthly_file=interim_dir / f"{gas}_gmnhsh-mean_monthly.nc",
                gmnhsh_mean_annual_file=interim_dir / f"{gas}_gmnhsh-mean_annual.nc",
            )
        )

    return out

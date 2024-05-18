"""
Create config for writing input4MIPs files
"""

from __future__ import annotations

from pathlib import Path

from local.config.write_input4mips import WriteInput4MIPsConfig


def create_write_input4mips_config(
    gases: tuple[str, ...],
    start_year: int,
    end_year: int,
) -> list[WriteInput4MIPsConfig]:
    """
    Create configuration for writing input4MIPs data

    Parameters
    ----------
    gases
        Gases for which to create the configuration

    start_year
        Start year for input4MIPs output

    end_year
        End year for input4MIPs output

    Returns
    -------
        Created configuration
    """
    input4mips_out_dir = Path("data/processed/esgf-ready")

    return [
        WriteInput4MIPsConfig(
            step_config_id=gas,
            gas=gas,
            input4mips_out_dir=input4mips_out_dir,
            complete_file=input4mips_out_dir / f"{gas}_input4MIPs_esgf-ready.complete",
            start_year=start_year,
            end_year=end_year,
        )
        for gas in gases
    ]

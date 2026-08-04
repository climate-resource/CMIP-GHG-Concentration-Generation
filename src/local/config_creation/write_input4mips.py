"""
Create config for writing input4MIPs files
"""

from __future__ import annotations

from pathlib import Path

from local.config.write_input4mips import WriteInput4MIPsConfig


def create_write_input4mips_config(  # noqa: PLR0913
    gases: tuple[str, ...],
    start_year: int,
    end_year: int,
    input4mips_cvs_source_id: str,
    input4mips_cvs_cv_source: str,
    gases_with_satellite_data: tuple[str, ...] = (),
    satellite_fit: str | None = None,
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

    input4mips_cvs_source_id
        Source ID to use to write the input4MIPs files

    input4mips_cvs_cv_source
        Source from which to retrieve the input4MIPs CVs

    gases_with_satellite_data
        Gases for which satellite data was included on top of the ground-based
        observational network. Their output file name(s) get an
        ``_SAT_{satellite_fit}`` suffix so they can be told apart from a run
        without satellite data.

    satellite_fit
        Fit used for the satellite data in ``gases_with_satellite_data``
        (e.g. ``"LINEAR_FIT"``). Only used for gases in ``gases_with_satellite_data``.

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
            complete_file_check_data=input4mips_out_dir / f"{gas}_input4MIPs_check-data.complete",
            complete_file=input4mips_out_dir / f"{gas}_input4MIPs_esgf-ready.complete",
            start_year=start_year,
            end_year=end_year,
            input4mips_cvs_source_id=input4mips_cvs_source_id,
            input4mips_cvs_cv_source=input4mips_cvs_cv_source,
            output_filename_suffix=f"SAT_{satellite_fit}" if gas in gases_with_satellite_data else None,
        )
        for gas in gases
    ]

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
    gases_satellite_fit: dict[str, str] | None = None,
    comment: str | None = None,
    activity_id: str | None = None,
    extra_metadata_by_gas: dict[str, dict[str, str]] | None = None,
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
        ``_SAT_{fit}`` suffix so they can be told apart from a run
        without satellite data.

    gases_satellite_fit
        Fit used for the satellite data, per gas in ``gases_with_satellite_data``
        (e.g. ``{"co2": "LINEAR_FIT"}``). Only used for gases in ``gases_with_satellite_data``.

    comment
        Value for the ``comment`` attribute of the output files.
        If ``None``, the notebook's default comment is used.

    activity_id
        Activity ID to write into the output files.
        If ``None``, the default (``input4MIPs``) is used.

    extra_metadata_by_gas
        Extra global attributes to add to the output files, per gas.
        Gases that are not in the dictionary get no extra attributes.

    Returns
    -------
        Created configuration
    """
    input4mips_out_dir = Path("data/processed/esgf-ready")

    if gases_satellite_fit is None:
        gases_satellite_fit = {}

    if extra_metadata_by_gas is None:
        extra_metadata_by_gas = {}

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
            output_filename_suffix=f"SAT_{gases_satellite_fit[gas]}"
            if gas in gases_with_satellite_data
            else None,
            comment=comment,
            activity_id=activity_id,
            extra_metadata=extra_metadata_by_gas.get(gas),
        )
        for gas in gases
    ]

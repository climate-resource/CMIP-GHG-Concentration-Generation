"""
Raw data processing
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS_REQUIRED_FOR_BINNING: tuple[str, ...] = (
    "gas",
    "year",
    "month",
    "latitude",
    "longitude",
    "value",
    "unit",
    "network",
    "station",
    "measurement_method",
)
"""Columns that are required for binning station data"""


def check_processed_data_columns_for_spatial_binning(indf: pd.DataFrame) -> None:
    """
    Check that the processed data has all the columns needed for spatial binning

    Parameters
    ----------
    indf
        :obj:`pd.DataFrame` to check

    Raises
    ------
    AssertionError
        Required columns are missing.
    """
    missing_latitudinal_mean_cols = set(COLUMNS_REQUIRED_FOR_BINNING) - set(indf.columns)
    if missing_latitudinal_mean_cols:
        msg = f"{missing_latitudinal_mean_cols=}"
        raise AssertionError(msg)


def read_and_check_binning_columns(infile: Path) -> pd.DataFrame:
    """
    Read a file, checking the binning columns before returning

    Parameters
    ----------
    infile
        Path from which to read the data

    Returns
    -------
        Loaded data, assuming that the checks pass
    """
    out = pd.read_csv(infile)
    check_processed_data_columns_for_spatial_binning(out)

    return out

"""
Interpolation of binned data
"""

from __future__ import annotations

import pandas as pd


def check_data_columns_for_binned_data_interpolation(indf: pd.DataFrame) -> None:
    missing = set(indf.columns).difference(
        {"gas", "lat_bin", "lon_bin", "month", "unit", "value", "year"}
    )
    if missing:
        msg = f"Missing required columns: {missing=}"
        raise AssertionError(msg)

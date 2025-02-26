"""
Harmonisation helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_harmonised_timeseries(
    ints: pd.DataFrame, harm_value: float, harm_units: str, harm_year: int, n_transition_years: int
) -> pd.DataFrame:
    """
    Harmonise a timeseries to a given value

    Uses a very basic declining offset harmonisation
    """
    ints_unit = ints.index.get_level_values("unit").unique()
    if len(ints_unit) > 1:
        raise AssertionError
    ints_unit = ints_unit[0]

    if ints_unit != harm_units:
        raise NotImplementedError

    delta = harm_value - ints.loc[:, harm_year]

    delta_to_add = ints.loc[:, :harm_year] * 0.0
    delta_to_add.loc[:, harm_year] = delta_to_add.loc[:, harm_year].add(delta)
    delta_to_add.loc[:, harm_year - n_transition_years + 1 : harm_year - 1] = np.nan
    # Linearly interpolate from zero to the offset value
    delta_to_add = delta_to_add.T.interpolate("index").T

    res = ints + delta_to_add
    # Only keep until one year before the harmonisation value,
    # as the harmonisation value is in the timeseries with which
    # we're harmonising.
    res = res.loc[:, : harm_year - 1]

    return res

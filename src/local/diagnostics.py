"""
Helpers for saving small, comparable diagnostics about a pipeline run

The goal is to be able to run the ``calculate_{gas}_monthly_fifteen_degree_pieces``
notebooks once without satellite data and once (or more) with satellite data
(potentially with different fits) and then compare the results,
without the different runs' diagnostics overwriting each other.

To do this, every diagnostics file this module writes is named using
:func:`get_satellite_suffix`, which mirrors the ``output_filename_suffix``
convention already used for the final input4MIPs files
(see ``local.config_creation.write_input4mips.create_write_input4mips_config``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
import yaml

_SCALED_SAT_DATE_RANGE_RE = re.compile(r"^(?P<start>\d{6})_(?P<end>\d{6})-")


def get_satellite_data_year_range(scaled_data_path: Path) -> tuple[int, int]:
    """
    Get the (start year, end year) covered by a scaled satellite data file, from its filename

    Every fit's raw file starts with the same ``"{start:%Y%m}_{end:%Y%m}-..."``
    prefix (e.g. ``"200301_202312-C3S-L3_..."``), so this works regardless of
    ``include_satellite_data``/``satellite_fit`` - i.e. it gives the same
    reference period whether or not satellite data is actually switched on
    for this run, which is what lets diagnostics scoped to "the years
    satellite data covers" be compared apples-to-apples across a `nosat` run
    and a `SAT_{fit}` run.

    Parameters
    ----------
    scaled_data_path
        Path to a scaled satellite data file (``config_process_scaled_sat_data.scaled_data_path``)

    Returns
    -------
        Start and end year covered by the file
    """
    match = _SCALED_SAT_DATE_RANGE_RE.match(scaled_data_path.name)
    if not match:
        msg = f"Could not parse a date range from {scaled_data_path.name}"
        raise ValueError(msg)

    return int(match.group("start")[:4]), int(match.group("end")[:4])


def get_satellite_suffix(include_satellite_data: bool, satellite_fit: str | None) -> str:
    """
    Get the suffix used to distinguish diagnostics produced with different satellite-data configurations

    Parameters
    ----------
    include_satellite_data
        Whether satellite data was included on top of the ground-based observational network

    satellite_fit
        Fit used for the satellite data (e.g. ``"LINEAR_FIT"``).
        Only used if ``include_satellite_data`` is ``True``.

    Returns
    -------
        ``"nosat"`` if ``include_satellite_data`` is ``False``,
        otherwise ``f"SAT_{satellite_fit}"``.
    """
    if not include_satellite_data:
        return "nosat"

    if not satellite_fit:
        msg = "satellite_fit must be provided if include_satellite_data is True"
        raise ValueError(msg)

    return f"SAT_{satellite_fit}"


def diagnostics_file_stem(gas: str, piece: str, suffix: str) -> str:
    """
    Build the common file stem used for a diagnostics file

    Parameters
    ----------
    gas
        Gas the diagnostics apply to

    piece
        Short label for the pipeline piece the diagnostics come from,
        e.g. ``"obs-network"``, ``"lat-gradient-extend"``

    suffix
        Satellite-data suffix, as returned by :func:`get_satellite_suffix`

    Returns
    -------
        File stem (no extension)
    """
    return f"{gas}_diagnostics_{piece}__{suffix}"


def record_check(
    name: str,
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    rtol: float | None = None,
    atol: float | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Compute and record the actual metric behind a correctness check

    This is meant to be used alongside (not instead of) the existing
    ``np.testing.assert_allclose`` calls that are already in the pipeline:
    the assert still raises if the check fails, this just also captures
    the actual numbers involved so they can be compared across runs
    instead of being silently discarded once the assert passes.

    Parameters
    ----------
    name
        Short, unique name for the check, e.g. ``"neem"``, ``"law_dome"``

    actual
        Actual (reconstructed) values

    expected
        Expected (reference) values

    rtol
        Relative tolerance used for the corresponding ``assert_allclose`` call, if any

    atol
        Absolute tolerance used for the corresponding ``assert_allclose`` call, if any

    description
        Human-readable description of what this check compares

    Returns
    -------
        Dictionary summarising the check, suitable for dumping to YAML
    """
    actual_arr = np.atleast_1d(np.asarray(actual, dtype=float))
    expected_arr = np.atleast_1d(np.asarray(expected, dtype=float))

    abs_diff = np.abs(actual_arr - expected_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(expected_arr != 0, abs_diff / np.abs(expected_arr), np.nan)

    max_rel_diff = float(np.nanmax(rel_diff)) if not np.all(np.isnan(rel_diff)) else None

    return {
        "name": name,
        "description": description,
        "n_points": int(actual_arr.size),
        "max_abs_diff": float(np.nanmax(abs_diff)),
        "mean_abs_diff": float(np.nanmean(abs_diff)),
        "max_rel_diff": max_rel_diff,
        "rtol_used": rtol,
        "atol_used": atol,
    }


def linear_regression_r2(x: npt.ArrayLike, y: npt.ArrayLike, m: float, c: float) -> float:
    """
    Calculate the R^2 of a fitted ``y = m * x + c`` linear regression

    Parameters
    ----------
    x
        x-values used in the regression

    y
        y-values used in the regression

    m
        Fitted gradient

    c
        Fitted intercept

    Returns
    -------
        R^2 of the fit
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    y_pred = m * x_arr + c
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)

    if ss_tot == 0:
        return float("nan")

    return float(1 - ss_res / ss_tot)


def explained_variance_ratio(principal_components: xr.DataArray, dim: str = "year") -> xr.DataArray:
    """
    Recover the singular value (and explained variance ratio) behind each EOF

    ``local.latitudinal_gradient.calculate_eofs_pcs`` and
    ``local.seasonality.calculate_seasonality_change_eofs_pcs`` compute
    an SVD (``U, D, Vh``) and only keep ``principal_components = U @ diag(D)``
    and ``eofs = Vh.T``. Since the columns of ``U`` are orthonormal,
    the singular values ``D`` can be recovered exactly as the column-wise
    norm of ``principal_components``, without needing to change those functions
    or redo the SVD.

    Parameters
    ----------
    principal_components
        Principal components, as returned by ``calculate_eofs_pcs`` /
        ``calculate_seasonality_change_eofs_pcs`` (i.e. the *full* set,
        before truncating to the number of EOFs actually used downstream)

    dim
        Dimension over which the principal components vary
        (``"year"`` for both the latitudinal gradient and seasonality change)

    Returns
    -------
        Explained variance ratio (fraction of total variance) for each EOF
    """
    pcs = principal_components.pint.dequantify()
    singular_values = np.sqrt((pcs**2).sum(dim=dim))
    variance = singular_values**2

    return (variance / variance.sum()).rename("explained_variance_ratio")


def save_yaml_diagnostics(path: Path, **diagnostics: Any) -> None:
    """
    Save a dictionary of scalar/small diagnostics to a YAML file

    Parameters
    ----------
    path
        Path to save the diagnostics to

    **diagnostics
        Diagnostics to save
    """
    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w") as fh:
        yaml.safe_dump(diagnostics, fh, sort_keys=False, default_flow_style=False)


def load_yaml_diagnostics(path: Path) -> dict[str, Any]:
    """
    Load a dictionary of scalar/small diagnostics from a YAML file

    Parameters
    ----------
    path
        Path to load the diagnostics from

    Returns
    -------
        Loaded diagnostics
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


def save_nc_diagnostics(path: Path, dataset: xr.Dataset) -> None:
    """
    Save an :obj:`xr.Dataset` of array-valued diagnostics to a NetCDF file

    Parameters
    ----------
    path
        Path to save the diagnostics to

    dataset
        Dataset to save. Anything with pint units should already be
        dequantified (as we do for all our other saved NetCDF files).
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(path)

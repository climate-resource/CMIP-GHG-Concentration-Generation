"""
Spatial manipulation of xarray objects
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def calculate_area_weighted_mean_latitude_only(
    inp: xr.DataArray,
    variables: list[str],
    bounds_dim_name: str = "bounds",
    lat_name: str = "lat",
    lat_bounds_name: str = "lat_bounds",
) -> xr.Dataset:
    """
    Calculate an area-weighted mean based on only latitude information

    This assumes that the data applies to the entire cell
    and is constant across the cell,
    hence we're effectively doing a weighted integral
    of a piecewise-constant function,
    rather than a weighted sum
    (which is what pure cos-weighting implies).

    See :footcite:t:`kelly_savric_2020_computation`

    @article{kelly_savric_2020_computation,
        author = {Kelly, Kevin and Šavrič, Bojan},
        title = {
            Area and volume computation of longitude-latitude grids and three-dimensional meshes
        },
        journal = {Transactions in GIS},
        volume = {25},
        number = {1},
        pages = {6-24},
        doi = {https://doi.org/10.1111/tgis.12636},
        url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/tgis.12636},
        eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/tgis.12636},
        abstract = {
            Abstract Longitude-latitude grids are commonly used for surface analyses
            and data storage in GIS. For volumetric analyses,
            three-dimensional meshes perpendicularly raised above or below the gridded surface are applied.
            Since grids and meshes are defined with geographic coordinates,
            they are not equal area or volume due to convergence of the meridians and radii.
            This article compiles and presents known geodetic considerations
            and relevant formulae needed for longitude-latitude grid and mesh analyses in GIS.
            The effect of neglecting these considerations is demonstrated on area
            and volume calculations of ecological marine units.
        },
        year = {2021}
    }

    Parameters
    ----------
    inp
        :obj:`xr.Dataset` to process

    variables
        Variables of which to calculate the area-mean

    bounds_dim_name
        Name of the dimension which defines bounds

    lat_name
        Name of the latitude dimension

    lat_bounds_name
        Name of the latitude bounds variable

    Returns
    -------
        :obj:`xr.Dataset` with area-weighted mean of ``variables``
    """
    lat_bnds = inp[lat_bounds_name].pint.to("radian")

    # The weights are effectively:
    # int_bl^bu r cos(theta) dphi r dtheta = int_bl^bu r^2 cos(theta) dtheta dphi
    # As they are constants, r^2 and longitude factors drop out in area weights.
    # (You would have to be more careful with longitudes if on a non-uniform grid).
    # When you evaluate the integral, you hence get that the weights are proportional to:
    # int_bl^bu cos(theta) dtheta = sin(bu) - sin(bl).
    # This is what we evaluate below.
    area_weighting = np.sin(lat_bnds).diff(dim=bounds_dim_name).squeeze()

    area_weighted_mean = (inp[variables] * area_weighting).sum(lat_name) / area_weighting.sum(lat_name)

    # May need to allow dependency injection in future here.
    keys_to_check = list(inp.data_vars.keys()) + list(inp.coords.keys())
    other_stuff = [v for v in keys_to_check if v not in variables]
    out = xr.merge([area_weighted_mean, inp[other_stuff]])

    return out


def calculate_cos_lat_weighted_mean_latitude_only(
    inda: xr.DataArray,
    lat_name: str = "lat",
) -> xr.DataArray:
    """
    Calculate cos of latitude-weighted mean

    This is just a simple, cos of latitude-weighted mean of the input data.
    Implicitly, this assumes that the data only applies to the point it sits on,
    in contrast to {py:func}`calculate_area_weighted_mean_latitude_only`,
    which implicitly assumes that the data applies to the entire cell
    (and some other things,
    see the docstring of {py:func}`calculate_area_weighted_mean_latitude_only`).

    Parameters
    ----------
    inda
        Input data on which to calculate the mean

    lat_name
        Name of the latitudinal dimension in ``inda``

    Returns
    -------
        Cos of latitude-weighted, latitudinal mean of ``inda``
    """
    weights = np.cos(np.deg2rad(inda[lat_name]))
    weights.name = "weights"

    return inda.weighted(weights=weights).mean(lat_name)


def calculate_global_mean_from_lon_mean(inda: xr.DataArray) -> xr.DataArray:
    """
    Calculate global-mean data from data which has already had a longitudinal mean applied.

    In other words, we assume that the data is on a latitudinal grid
    (with perhaps other non-spatial elements too).
    We also assume that the data applies to points, rather than areas.
    Hence we use {py:func}`calculate_cos_lat_weighted_mean_latitude_only`
    rather than {py:func}`calculate_area_weighted_mean_latitude_only`.

    Parameters
    ----------
    inda
        Input data

    Returns
    -------
        Global-mean of ``inda``.
    """
    return calculate_cos_lat_weighted_mean_latitude_only(inda)

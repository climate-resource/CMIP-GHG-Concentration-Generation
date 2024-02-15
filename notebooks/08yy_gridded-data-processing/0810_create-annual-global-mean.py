# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Create global-/hemispheric- and annual-means
#
# These are summary-products of our gridded data.
#
# All derived from 15 degree latitudinal grid, see Figure 1 of Meinshausen et al., 2017 https://gmd.copernicus.org/articles/10/2057/2017/gmd-10-2057-2017.pdf

# %% [markdown]
# ## Imports

# %%
from collections.abc import Callable

import carpet_concentrations.xarray_utils
import cf_xarray.units
import matplotlib.pyplot as plt
import pint_xarray
import xarray as xr

# TODO: move this function
from carpet_concentrations.input4MIPs.dataset import add_time_bounds
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "gridded_data_processing"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %%
config_grid = get_config_for_step_id(config=config, step="grid", step_config_id="only")

# %% [markdown]
# ## Action

# %% [markdown]
# ### Set-up unit registry

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

# %% [markdown]
# ## Load gridded data

# %%
raw_gridded = xr.open_dataset(
    config_grid.processed_data_file, use_cftime=True
).pint.quantify()
raw_gridded

# %%
bounded_gridded = raw_gridded.cf.add_bounds("lat").pint.quantify(
    lat_bounds="degrees_north", lat="degrees_north"
)
bounded_gridded


# TODO: speak with Paul about this. This sector business seems like a pain for
# all involved.
def get_gmnhsh_data(
    inp: xr.Dataset,
    variables: list[str],
    output_spatial_dim_name: str = "sector",
    calculate_weighted_area_mean_latitude_only: Callable[
        [xr.Dataset, list[str]], xr.Dataset
    ]
    | None = None,
) -> xr.Dataset:
    """
    Get global-mean and hemispheric data
    """
    if calculate_weighted_area_mean_latitude_only is None:
        calculate_weighted_area_mean_latitude_only = (
            carpet_concentrations.xarray_utils.calculate_weighted_area_mean_latitude_only
        )

    nh = inp["lat"] >= 0

    means = []
    ids = []
    lat_bounds = []
    for idx, (name, inp_selected) in enumerate(
        [
            ("Global", inp),
            ("Northern Hemisphere", inp.loc[{"lat": nh}]),
            ("Southern Hemisphere", inp.loc[{"lat": ~nh}]),
        ]
    ):
        mean = (
            calculate_weighted_area_mean_latitude_only(inp_selected, variables)
            .expand_dims({output_spatial_dim_name: [idx]})
            .drop(["lat_bounds", "lat"])
        )
        means.append(mean)

        lat_bounds_m = inp_selected["lat_bounds"].data.to("degrees_north").m
        lat_min = float(lat_bounds_m.min())
        lat_max = float(lat_bounds_m.max())

        ids.append(f"{idx}: {name}")
        lat_bounds.append(f"{idx}: {lat_min:.1f}, {lat_max:.1f}")

    out: xr.Dataset = xr.concat(means, dim=output_spatial_dim_name).cf.add_bounds(
        output_spatial_dim_name
    )

    for v in variables:
        out[v].attrs.update(inp[v].attrs)

    out[output_spatial_dim_name].attrs.update(
        {
            "long_name": "sector",
            "ids": "; ".join(ids),
            "lat_bounds": "; ".join(lat_bounds),
        }
    )

    return out


# %%
gmnhsh_data = get_gmnhsh_data(bounded_gridded, list(raw_gridded.data_vars.keys()))
for vda in gmnhsh_data.data_vars.values():
    vda.sel(region="World", scenario="historical").plot(hue="sector", alpha=0.7)
    plt.show()

config_step.processed_data_file_global_hemispheric_means.parent.mkdir(
    exist_ok=True, parents=True
)
gmnhsh_data.to_netcdf(config_step.processed_data_file_global_hemispheric_means)
print(config_step.processed_data_file_global_hemispheric_means)
gmnhsh_data

# %%
gmnhsh_data_time_bnds = add_time_bounds(gmnhsh_data.copy(), monthly_time_bounds=True)
time_weights = xr.DataArray(
    [
        v.days
        for v in gmnhsh_data_time_bnds["time_bounds"].data[:, 1]
        - gmnhsh_data_time_bnds["time_bounds"].data[:, 0]
    ],
    coords=[("time", gmnhsh_data_time_bnds["time"].values)],
    dims=("time",),
    name="time_weights",
)
print(time_weights)

time_name = "time"
gmnhsh_data_annual_mean = (
    (gmnhsh_data * time_weights)
    .groupby("time.year")
    .sum(
        time_name,
        # ensure we only take averages over full years
        min_count=12,
    )
    / time_weights.groupby("time.year").sum(time_name)
).dropna("year")

for vda in gmnhsh_data_annual_mean.data_vars.values():
    vda.sel(region="World", scenario="historical").plot(hue="sector", alpha=0.7)
    plt.show()

config_step.processed_data_file_global_hemispheric_annual_means.parent.mkdir(
    exist_ok=True, parents=True
)
gmnhsh_data_annual_mean.to_netcdf(
    config_step.processed_data_file_global_hemispheric_annual_means
)
print(config_step.processed_data_file_global_hemispheric_annual_means)

gmnhsh_data_annual_mean

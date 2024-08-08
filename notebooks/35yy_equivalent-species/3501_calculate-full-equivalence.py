# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calculate full equivalence
#
# Here we calculate a full equivalent dataset.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cf_xarray.units
import pint
import pint_xarray
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

Q = pint.get_application_registry().Quantity  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "crunch_equivalent_species"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "cfc11eq"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)
gas_raw = config_step.gas.replace("eq", "")

config_grid_crunching_included_gases = [
    get_config_for_step_id(
        config=config,
        step="crunch_grids",
        step_config_id=gas,
    )
    for gas in config_step.equivalent_component_gases
]


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load comparison data

# %%
raw_global_mean_monthly: xr.DataArray = xr.load_dataarray(  # type: ignore
    get_config_for_step_id(
        config=config,
        step="crunch_grids",
        step_config_id=gas_raw,
    ).global_mean_monthly_file
).pint.quantify()
raw_global_mean_monthly

# %% [markdown]
# ### Calculate equivalents

# %%
# TODO: update all of these based on AR6
RADIATIVE_EFFICIENCIES: dict[str, pint.UnitRegistry.Quantity] = {
    "c2f6": Q(0.25, "W / m^2 / ppb"),
    "c3f8": Q(0.28, "W / m^2 / ppb"),
    "c4f10": Q(0.36, "W / m^2 / ppb"),
    "c5f12": Q(0.41, "W / m^2 / ppb"),
    "c6f14": Q(0.44, "W / m^2 / ppb"),
    "c7f16": Q(0.5, "W / m^2 / ppb"),
    "c8f18": Q(0.55, "W / m^2 / ppb"),
    "cc4f8": Q(0.32, "W / m^2 / ppb"),
    "ccl4": Q(0.017, "W / m^2 / ppb"),
    "cf4": Q(0.09, "W / m^2 / ppb"),
    "cfc11": Q(0.26, "W / m^2 / ppb"),
    "cfc113": Q(0.3, "W / m^2 / ppb"),
    "cfc114": Q(0.31, "W / m^2 / ppb"),
    "cfc115": Q(0.2, "W / m^2 / ppb"),
    "cfc12": Q(0.32, "W / m^2 / ppb"),
    "ch2cl2": Q(0.03, "W / m^2 / ppb"),
    "ch3br": Q(0.004, "W / m^2 / ppb"),
    "ch3ccl3": Q(0.07, "W / m^2 / ppb"),
    "ch3cl": Q(0.01, "W / m^2 / ppb"),
    "chcl3": Q(0.08, "W / m^2 / ppb"),
    "halon1211": Q(0.29, "W / m^2 / ppb"),
    "halon1301": Q(0.3, "W / m^2 / ppb"),
    "halon2402": Q(0.31, "W / m^2 / ppb"),
    "hcfc141b": Q(0.16, "W / m^2 / ppb"),
    "hcfc142b": Q(0.19, "W / m^2 / ppb"),
    "hcfc22": Q(0.21, "W / m^2 / ppb"),
    "hfc125": Q(0.23, "W / m^2 / ppb"),
    "hfc134a": Q(0.16, "W / m^2 / ppb"),
    "hfc143a": Q(0.16, "W / m^2 / ppb"),
    "hfc152a": Q(0.1, "W / m^2 / ppb"),
    "hfc227ea": Q(0.26, "W / m^2 / ppb"),
    "hfc23": Q(0.18, "W / m^2 / ppb"),
    "hfc236fa": Q(0.24, "W / m^2 / ppb"),
    "hfc245fa": Q(0.24, "W / m^2 / ppb"),
    "hfc32": Q(0.11, "W / m^2 / ppb"),
    "hfc365mfc": Q(0.22, "W / m^2 / ppb"),
    "hfc4310mee": Q(0.42, "W / m^2 / ppb"),
    "nf3": Q(0.2, "W / m^2 / ppb"),
    "sf6": Q(0.57, "W / m^2 / ppb"),
    "so2f2": Q(0.2, "W / m^2 / ppb"),
}

# %%
equivalents = {}
for key, attr_to_grab in (
    ("fifteen_degree", "fifteen_degree_monthly_file"),
    # ("half_degree", "half_degree_monthly_file"),
    ("global_mean_monthly", "global_mean_monthly_file"),
    ("hemispheric_mean_monthly", "hemispheric_mean_monthly_file"),
    ("global_mean_annual_mean", "global_mean_annual_mean_file"),
    ("hemispheric_mean_annual_mean", "hemispheric_mean_annual_mean_file"),
):
    print(f"Crunching {key}")
    total_erf_set = False
    included_species = []

    for crunch_gas_config in config_grid_crunching_included_gases:
        loaded = xr.load_dataarray(  # type: ignore
            getattr(crunch_gas_config, attr_to_grab)
        ).pint.quantify()

        if loaded.name != crunch_gas_config.gas:
            raise AssertionError

        loaded_erf = (loaded * RADIATIVE_EFFICIENCIES[crunch_gas_config.gas]).pint.to(
            "W / m^2"
        )

        print(f"Adding {loaded.name}")
        included_species.append(loaded.name)

        if not total_erf_set:
            total_erf_set = True
            total_erf = loaded_erf
        else:
            total_erf += loaded_erf.sel(year=total_erf["year"])

    total = (total_erf / RADIATIVE_EFFICIENCIES[gas_raw]).pint.to(
        raw_global_mean_monthly.data.units
    )
    total.name = config_step.gas
    total.attrs[
        "commment"
    ] = f"{config_step.gas} is the equivalent of {', '.join(included_species)}"
    equivalents[key] = total
    # Set metadata about components etc. here
    print()

# %%
local.xarray_time.convert_year_month_to_time(
    equivalents["global_mean_monthly"], calendar="proleptic_gregorian"
).plot.line()
local.xarray_time.convert_year_month_to_time(
    raw_global_mean_monthly, calendar="proleptic_gregorian"
).plot.line()

# %% [markdown]
# ### Save

# %%
config_step.fifteen_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["fifteen_degree"].pint.dequantify().to_netcdf(
    config_step.fifteen_degree_monthly_file
)
equivalents["fifteen_degree"]

# # %%
# config_step.half_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
# equivalents["half_degree"].pint.dequantify().to_netcdf(
#     config_step.half_degree_monthly_file
# )
# equivalents["half_degree"]

# %%
config_step.global_mean_monthly_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["global_mean_monthly"].pint.dequantify().to_netcdf(
    config_step.global_mean_monthly_file
)
equivalents["global_mean_monthly"]

# %%
config_step.hemispheric_mean_monthly_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["hemispheric_mean_monthly"].pint.dequantify().to_netcdf(
    config_step.hemispheric_mean_monthly_file
)
equivalents["hemispheric_mean_monthly"]

# %%
config_step.global_mean_annual_mean_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["global_mean_annual_mean"].pint.dequantify().to_netcdf(
    config_step.global_mean_annual_mean_file
)
equivalents["global_mean_annual_mean"]

# %%
config_step.hemispheric_mean_annual_mean_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["hemispheric_mean_annual_mean"].pint.dequantify().to_netcdf(
    config_step.hemispheric_mean_annual_mean_file
)
equivalents["hemispheric_mean_annual_mean"]

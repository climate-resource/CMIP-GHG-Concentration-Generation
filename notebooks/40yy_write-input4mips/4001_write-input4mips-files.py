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
# # Write input4MIPs files
#
# Here we write our input4MIPs files for our four
# different gridded data products:
#
# - 15&deg; latitudinal, monthly
# - 0.5&deg; latitudinal, monthly
# - global-, northern hemisphere-mean, southern-hemisphere mean, monthly
# - global-, northern hemisphere-mean, southern-hemisphere mean, annual-mean

# %% [markdown]
# ## Imports

# %%
import datetime
from functools import partial

import cf_xarray.units
import cftime
import numpy as np
import pint_xarray
import tqdm.autonotebook as tqdman
import xarray as xr
from input4mips_validation.dataset import Input4MIPsDataset
from input4mips_validation.metadata import (
    Input4MIPsMetadata,
    Input4MIPsMetadataOptional,
)
from input4mips_validation.xarray_helpers import add_time_bounds
from pydoit_nb.checklist import generate_directory_checklist
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

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "write_input4mips"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "sf6"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_grid_crunching = get_config_for_step_id(
    config=config,
    step="crunch_grids",
    step_config_id=config_step.gas,
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
fifteen_degree_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_grid_crunching.fifteen_degree_monthly_file
).pint.quantify()

half_degree_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_grid_crunching.half_degree_monthly_file
).pint.quantify()

gmnhsh_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_grid_crunching.gmnhsh_mean_monthly_file
).pint.quantify()

gmnhsh_annual_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_grid_crunching.gmnhsh_mean_annual_file
).pint.quantify()


# %% [markdown]
# ### Check time axis


# %%
def chop_time_axis(inp: xr.DataArray) -> xr.DataArray:
    """
    Chop the time axis to our desired time axis
    """
    res = inp.sel(year=range(config_step.start_year, config_step.end_year + 1))

    if res.isnull().any():
        raise AssertionError

    return res


# %%
fifteen_degree_data_raw_chopped = chop_time_axis(fifteen_degree_data_raw)
half_degree_data_raw_chopped = chop_time_axis(half_degree_data_raw)
gmnhsh_data_raw_chopped = chop_time_axis(gmnhsh_data_raw)
gmnhsh_annual_data_raw_chopped = chop_time_axis(gmnhsh_annual_data_raw)


# %% [markdown]
# ### Convert everything to a time axis


# %%
def get_displayable_dataarray(inp: xr.DataArray) -> xr.DataArray:
    """
    Get a :obj:`xr.DataArray` which we can dispaly

    There is some bug in xarray's HTML representation which
    means this doesn't work with a proleptic_gregorian calendar.
    """
    res = inp.copy()
    res["time"] = np.array(
        [
            cftime.datetime(v.year, v.month, v.day, v.hour, calendar="standard")
            for v in inp["time"].values
        ]
    )

    return res


# %%
day = 15
fifteen_degree_data = local.xarray_time.convert_year_month_to_time(
    fifteen_degree_data_raw_chopped, day=day
)
half_degree_data = local.xarray_time.convert_year_month_to_time(
    half_degree_data_raw_chopped, day=day
)
gmnhsh_data = local.xarray_time.convert_year_month_to_time(
    gmnhsh_data_raw_chopped, day=day
)
get_displayable_dataarray(fifteen_degree_data)

# %%
gmnhsh_annual_data = local.xarray_time.convert_year_to_time(
    gmnhsh_annual_data_raw_chopped,
    month=7,
    day=2,
    hour=12,
    calendar="proleptic_gregorian",
)
get_displayable_dataarray(gmnhsh_annual_data)

# %% [markdown]
# ### Set common metadata

# %%
version = config.version

metadata_universal = dict(
    activity_id="input4MIPs",
    contact="zebedee.nicholls@climate-resource.com;malte.meinshausen@climate-resource.com",
    # institution="Climate Resource, Fitzroy, Victoria 3065, Australia",
    institution_id="CR",
    mip_era="CMIP6Plus",
    target_mip="CMIP",
    source_id=f"CR_hist-ghg-concs_{version}",
)

metadata_universal_optional: dict[str, str] = dict(
    # product="derived",
    # TODO: add support for this to input4mips-validation
    # further_info_url="https://github.com/climate-resource/CMIP-GHG-Concentration-Generation",
    # # TODO: check if there is a more exact grant agreement to refer to
    comment=(
        "[TBC which grant] Data produced by Climate Resource supported by funding "
        "from the CMIP IPO (Coupled Model Intercomparison Project International Project Office). "
        "This is an interim dataset, do not use in production."
    ),
    # TODO: add support for this to input4mips-validation
    # references="Meinshausen et al., 2017, GMD (https://doi.org/10.5194/gmd-10-2057-2017)",
)

# %% [markdown]
# ### Define variable renaming

# %%
gas_to_cmip_variable_renaming = {
    "co2": "mole_fraction_of_carbon_dioxide_in_air",
    "ch4": "mole_fraction_of_methane_in_air",
    "n2o": "mole_fraction_of_nitrous_oxide_in_air",
    "sf6": "mole_fraction_of_sulfur_hexafluoride_in_air",
}

# %% [markdown]
# ### Write files

# %%
config_step.input4mips_out_dir.mkdir(exist_ok=True, parents=True)

time_dimension = "time"
for dat_resolution, tmp_grid_name, yearly_time_bounds in tqdman.tqdm(
    [
        (fifteen_degree_data, "15_deg_lat", False),
        (half_degree_data, "05_deg_lat", False),
        (gmnhsh_data, "gmnhsh", False),
        (gmnhsh_annual_data, "gmnhsh", True),
    ],
    desc="Resolutions",
):
    grid_info = " x ".join(
        [f"{dat_resolution[v].size} ({v})" for v in dat_resolution.dims]
    )
    print(f"Processing {grid_info} grid")

    variable_name_raw = str(dat_resolution.name)
    variable_name_output = gas_to_cmip_variable_renaming[variable_name_raw]
    da_to_write = dat_resolution.to_dataset().rename_vars(
        {variable_name_raw: variable_name_output}
    )
    da_to_write["time"].encoding = {
        "calendar": "proleptic_gregorian",
        "units": "days since 1850-01-01",
    }
    # TODO: use inference again once I know how it is meant to work
    # metadata_inferred, metadata_inferred_optional = infer_metadata_from_dataset(
    #     da_to_write, scenario
    # )

    # metadata = Input4MIPsMetadata(
    #     **metadata_universal,
    #     **metadata_inferred,
    # )

    # metadata_optional = Input4MIPsMetadataOptional(
    #     **metadata_universal_optional,
    #     **metadata_inferred_optional,
    # )

    metadata = Input4MIPsMetadata(
        **metadata_universal,
        # Rules here make no sense to me,
        # can this be inferred from the data or only checked against it?
        grid_label=tmp_grid_name,
    )
    metadata_optional = Input4MIPsMetadataOptional(
        **metadata_universal_optional,
    )

    dimensions = tuple(str(v) for v in da_to_write[variable_name_output].dims)

    input4mips_ds = Input4MIPsDataset.from_raw_dataset(
        da_to_write,
        dimensions=dimensions,
        time_dimension=time_dimension,
        metadata=metadata,
        metadata_optional=metadata_optional,
        add_time_bounds=partial(
            add_time_bounds,
            monthly_time_bounds=not yearly_time_bounds,
            yearly_time_bounds=yearly_time_bounds,
        ),
    )
    print("Writing")
    written = input4mips_ds.write(root_data_dir=config_step.input4mips_out_dir)
    print(f"Wrote: {written.relative_to(config_step.input4mips_out_dir)}")
    print("")

checklist_path = generate_directory_checklist(config_step.input4mips_out_dir)
# Not sure if this is needed or not
with open(config_step.complete_file, "w") as fh:
    fh.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

checklist_path

# %%
config_step.input4mips_out_dir

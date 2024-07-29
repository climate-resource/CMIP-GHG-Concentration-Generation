# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
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
from attrs import evolve
from input4mips_validation.cvs.loading import load_cvs
from input4mips_validation.cvs.loading_raw import get_raw_cvs_loader
from input4mips_validation.dataset import Input4MIPsDataset
from input4mips_validation.dataset.metadata_data_producer_minimum import (
    Input4MIPsDatasetMetadataDataProducerMinimum,
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
step_config_id: str = "cfc11eq"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

if "eq" in config_step.gas:
    config_crunch_grids = get_config_for_step_id(
        config=config,
        step="crunch_equivalent_species",
        step_config_id=config_step.gas,
    )

else:
    config_crunch_grids = get_config_for_step_id(
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
    config_crunch_grids.fifteen_degree_monthly_file
).pint.quantify()

# half_degree_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
#     config_crunch_grids.half_degree_monthly_file
# ).pint.quantify()

gmnhsh_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.gmnhsh_mean_monthly_file
).pint.quantify()

gmnhsh_annual_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.gmnhsh_mean_annual_file
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
# half_degree_data_raw_chopped = chop_time_axis(half_degree_data_raw)
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
# half_degree_data = local.xarray_time.convert_year_month_to_time(
#     half_degree_data_raw_chopped, day=day
# )
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
version.replace(".", "-")

metadata_minimum = Input4MIPsDatasetMetadataDataProducerMinimum(
    grid_label="gn",
    nominal_resolution="10000 km",
    source_id=f"CR-CMIP-{version.replace('.', '-')}",
    target_mip="CMIP",
)
metadata_minimum

# %% [markdown]
# ### Define variable renaming

# %%
gas_to_cmip_variable_renaming = {
    "co2": "mole_fraction_of_carbon_dioxide_in_air",
    "ch4": "mole_fraction_of_methane_in_air",
    "n2o": "mole_fraction_of_nitrous_oxide_in_air",
    "c2f6": "mole_fraction_of_pfc116_in_air",
    "c3f8": "mole_fraction_of_pfc218_in_air",
    "c4f10": "mole_fraction_of_pfc3110_in_air",
    "c5f12": "mole_fraction_of_pfc4112_in_air",
    "c6f14": "mole_fraction_of_pfc5114_in_air",
    "c7f16": "mole_fraction_of_pfc6116_in_air",
    "c8f18": "mole_fraction_of_pfc7118_in_air",
    "cc4f8": "mole_fraction_of_pfc318_in_air",
    "ccl4": "mole_fraction_of_carbon_tetrachloride_in_air",
    "cf4": "mole_fraction_of_carbon_tetrafluoride_in_air",
    "cfc11": "mole_fraction_of_cfc11_in_air",
    "cfc113": "mole_fraction_of_cfc113_in_air",
    "cfc114": "mole_fraction_of_cfc114_in_air",
    "cfc115": "mole_fraction_of_cfc115_in_air",
    "cfc12": "mole_fraction_of_cfc12_in_air",
    "ch2cl2": "mole_fraction_of_dichloromethane_in_air",
    "ch3br": "mole_fraction_of_methyl_bromide_in_air",
    "ch3ccl3": "mole_fraction_of_hcc140a_in_air",
    "ch3cl": "mole_fraction_of_methyl_chloride_in_air",
    "chcl3": "mole_fraction_of_chloroform_in_air",
    "halon1211": "mole_fraction_of_halon1211_in_air",
    "halon1301": "mole_fraction_of_halon1301_in_air",
    "halon2402": "mole_fraction_of_halon2402_in_air",
    "hcfc141b": "mole_fraction_of_hcfc141b_in_air",
    "hcfc142b": "mole_fraction_of_hcfc142b_in_air",
    "hcfc22": "mole_fraction_of_hcfc22_in_air",
    "hfc125": "mole_fraction_of_hfc125_in_air",
    "hfc134a": "mole_fraction_of_hfc134a_in_air",
    "hfc143a": "mole_fraction_of_hfc143a_in_air",
    "hfc152a": "mole_fraction_of_hfc152a_in_air",
    "hfc227ea": "mole_fraction_of_hfc227ea_in_air",
    "hfc23": "mole_fraction_of_hfc23_in_air",
    "hfc236fa": "mole_fraction_of_hfc236fa_in_air",
    "hfc245fa": "mole_fraction_of_hfc245fa_in_air",
    "hfc32": "mole_fraction_of_hfc32_in_air",
    "hfc365mfc": "mole_fraction_of_hfc365mfc_in_air",
    "hfc4310mee": "mole_fraction_of_hfc4310mee_in_air",
    "nf3": "mole_fraction_of_nitrogen_trifluoride_in_air",
    "sf6": "mole_fraction_of_sulfur_hexafluoride_in_air",
    "so2f2": "mole_fraction_of_sulfuryl_fluoride_in_air",
    "cfc11eq": "mole_fraction_of_cfc11_eq_in_air",
    "cfc12eq": "mole_fraction_of_cfc12_eq_in_air",
    "hfc134aeq": "mole_fraction_of_hfc134a_eq_in_air",
}

# %% [markdown]
# ## Load CVs

# %%
raw_cvs_loader = get_raw_cvs_loader(
    "https://raw.githubusercontent.com/PCMDI/input4MIPs_CVs/main/CVs/"
)

# %%
cvs = load_cvs(raw_cvs_loader)
cvs.source_id_entries.source_ids

# %% [markdown]
# ### Write files

# %%
config_step.input4mips_out_dir.mkdir(exist_ok=True, parents=True)

time_dimension = "time"
for dat_resolution, tmp_grid_name, yearly_time_bounds in tqdman.tqdm(
    [
        (fifteen_degree_data, "15_deg_lat", False),
        # (half_degree_data, "05_deg_lat", False),
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
    ds_to_write = dat_resolution.to_dataset().rename_vars(
        {variable_name_raw: variable_name_output}
    )

    dimensions = tuple(str(v) for v in ds_to_write[variable_name_output].dims)
    print(f"{tmp_grid_name=}")
    print(f"{dimensions=}")

    ds_to_write["time"].encoding = {
        "calendar": "proleptic_gregorian",
        "units": "days since 1850-01-01",
        # Time has to be encoded as float
        # to ensure that non-integer days etc. can be handled
        # and the CF-checker doesn't complain.
        "dtype": np.dtypes.Float32DType,
    }

    if "lat" in dimensions:
        ds_to_write["lat"].encoding = {"dtype": np.dtypes.Float16DType}

    if "sector" in dimensions:
        ds_to_write["sector"].attrs["units"] = "1"

    input4mips_ds = Input4MIPsDataset.from_data_producer_minimum_information(
        data=ds_to_write,
        metadata_minimum=metadata_minimum,
        dimensions=dimensions,
        time_dimension=time_dimension,
        add_time_bounds=partial(
            add_time_bounds,
            monthly_time_bounds=not yearly_time_bounds,
            yearly_time_bounds=yearly_time_bounds,
        ),
        cvs=cvs,
        standard_and_or_long_names={
            variable_name_output: {"standard_name": variable_name_output},
            "sector": {
                "long_name": "ID for the region. This is legacy from CMIP6 and will be removed in future."
            },
        },
        dataset_category="GHGConcentrations",
        realm="atmos",
    )

    metadata_evolved = evolve(
        input4mips_ds.metadata,
        product="derived",
        comment=(
            "[TBC which grant] Data produced by Climate Resource supported by funding "
            "from the CMIP IPO (Coupled Model Intercomparison Project International Project Office). "
            "This is an interim dataset, do not use in production."
        ),
    )
    input4mips_ds = Input4MIPsDataset(
        data=input4mips_ds.data,
        metadata=metadata_evolved,
        cvs=cvs,
        non_input4mips_metadata=dict(
            references="Meinshausen et al., 2017, GMD (https://doi.org/10.5194/gmd-10-2057-2017)"
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

# %% [markdown]
# ## Validate the files

# %%
# !input4mips-validation --logging-level INFO_INDIVIDUAL_CHECK \
#     validate-tree {config_step.input4mips_out_dir} \
#     --cv-source "gh:main" \
#     --rglob-input "**/*{variable_name_output.replace('_', '-')}*/**/*.nc"

# %%
config_step.input4mips_out_dir

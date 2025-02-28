# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Write input4MIPs files
#
# Here we write our input4MIPs files for our six
# different gridded data products:
#
# - 15&deg; latitudinal, monthly
# - 0.5&deg; latitudinal, monthly
# - global-mean, monthly
# - northern hemisphere-mean, southern-hemisphere mean, monthly
# - global-mean, annual-mean
# - northern hemisphere-mean, southern-hemisphere mean, annual-mean

# %% [markdown]
# ## Imports

# %%
import datetime
import itertools
import json
from functools import partial
from pathlib import Path

import cf_xarray.units
import cftime
import numpy as np
import pint_xarray
import tqdm.autonotebook as tqdman
import xarray as xr
from attrs import evolve
from input4mips_validation.cvs.loading import load_cvs_known_loader
from input4mips_validation.cvs.loading_raw import get_raw_cvs_loader
from input4mips_validation.dataset import Input4MIPsDataset
from input4mips_validation.dataset.dataset import prepare_ds_and_get_frequency
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true
step: str = "write_input4mips"

# %% [markdown]
# ## Parameters

# %% editable=true tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "c4f10"  # config ID to select for this branch

# %% [markdown] editable=true
# ## Load config

# %% editable=true
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

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

global_mean_monthly_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.global_mean_monthly_file
).pint.quantify()

hemispheric_mean_monthly_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.hemispheric_mean_monthly_file
).pint.quantify()

global_mean_annual_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.global_mean_annual_mean_file
).pint.quantify()

hemispheric_mean_annual_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.hemispheric_mean_annual_mean_file
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
global_mean_monthly_data_raw_chopped = chop_time_axis(global_mean_monthly_data_raw)
hemispheric_mean_monthly_data_raw_chopped = chop_time_axis(hemispheric_mean_monthly_data_raw)
global_mean_annual_data_raw_chopped = chop_time_axis(global_mean_annual_data_raw)
hemispheric_mean_annual_data_raw_chopped = chop_time_axis(hemispheric_mean_annual_data_raw)


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
        [cftime.datetime(v.year, v.month, v.day, v.hour, calendar="standard") for v in inp["time"].values]
    )

    return res


# %%
day = 15
fifteen_degree_data = local.xarray_time.convert_year_month_to_time(fifteen_degree_data_raw_chopped, day=day)
# half_degree_data = local.xarray_time.convert_year_month_to_time(
#     half_degree_data_raw_chopped, day=day
# )
global_mean_monthly_data = local.xarray_time.convert_year_month_to_time(
    global_mean_monthly_data_raw_chopped, day=day
)
hemispheric_mean_monthly_data = local.xarray_time.convert_year_month_to_time(
    hemispheric_mean_monthly_data_raw_chopped, day=day
)
get_displayable_dataarray(fifteen_degree_data)

# %%
global_mean_annual_data = local.xarray_time.convert_year_to_time(
    global_mean_annual_data_raw_chopped,
    month=7,
    day=2,
    hour=12,
    calendar="proleptic_gregorian",
)
hemispheric_mean_annual_data = local.xarray_time.convert_year_to_time(
    hemispheric_mean_annual_data_raw_chopped,
    month=7,
    day=2,
    hour=12,
    calendar="proleptic_gregorian",
)
get_displayable_dataarray(global_mean_annual_data)

# %% [markdown]
# ### Set common metadata

# %%
version = config.version
version.replace(".", "-")

metadata_minimum_common = dict(
    source_id=config_step.input4mips_cvs_source_id,
    target_mip="CMIP",
)
metadata_minimum_common

# %%
import pandas as pd
import sqlite3

# %%
db_connection = sqlite3.connect("../../tmp.db")
sources = pd.read_sql("SELECT * FROM source", con=db_connection)
dependencies = pd.read_sql("SELECT * FROM dependencies", con=db_connection)
db_connection.close()

# %%
dependencies_gas = dependencies[dependencies["gas"] == config_step.gas]["short_name"].tolist()
if not dependencies_gas:
    raise AssertionError
    
dependencies_gas

# %%
gas_deps = sources[sources["short_name"].isin(dependencies_gas)].to_dict("records")

gas_deps.extend(
    (
        # TODO: insert this everywhere sooner
        # and make sure it comes through Zenodo records.
        {
            "short_name": "Meinshausen et al., GMD (2017)",
            "licence": "Paper, NA",
            "reference": (
                "Meinshausen, M., Vogel, E., ..., Wang, R. H. J., and Weiss, R.: "
                "Historical greenhouse gas concentrations for climate modelling (CMIP6), "
                "Geosci. Model Dev., 10, 2057-2116, https://doi.org/10.5194/gmd-10-2057-2017, 2017."
            ),
            "resource_type": "publication-article",
            "doi": "https://doi.org/10.5194/gmd-10-2057-2017",
            "url": "https://doi.org/10.5194/gmd-10-2057-2017",
        },
        {
            "short_name": "Nicholls et al., in-prep (2025)",
            "licence": "Paper, NA",
            "reference": (
                "Nicholls, Z., Meinshausen, M., Lewis, J., Pflueger, M., Menking, A., ...: "
                "Greenhouse gas concentrations for climate modelling (CMIP7), "
                "in-prep, 2025."
            ),
            "url": "https://github.com/climate-resource/CMIP-GHG-Concentration-Generation",
        } 
    )
)
gas_deps

# %%
# # TODO: replace this with generation of references throughout the workflow,
# # rather than from static file
# run_id = config_step.input4mips_out_dir.parents[2].name
# data_dir = config_step.input4mips_out_dir.parents[1]

# with open(data_dir / "raw" / "dependencies-by-gas.json") as fh:
#     all_gas_deps = json.load(fh)

# try:
#     gas_deps = all_gas_deps[config_step.gas]
# except KeyError:
#     # No deps yet, assume it came from SSP2-4.5
#     gas_deps = [
#         # TODO: move this earlier
#         {
#             "gas": config_step.gas,
#             "source": "Meinshausen et al., GMD (2020)",
#             "licence": "Paper, NA",
#             "reference": (
#                 "Meinshausen, M., Nicholls, Z. R. J., ..., Vollmer, M. K., and Wang, R. H. J.: "
#                 "The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500, "
#                 "Geosci. Model Dev., 13, 3571-3605, https://doi.org/10.5194/gmd-13-3571-2020, 2020."
#             ),
#             "doi": "https://doi.org/10.5194/gmd-10-2057-2017",
#             "url": "https://doi.org/10.5194/gmd-13-3571-2020",
#         }
#     ]


# gas_deps

# %%
# Order deps by reverse alphabetical order for now,
# can sort out order of priority when solving #62.
gas_deps = sorted(gas_deps, key=lambda v: v["short_name"])[::-1]
gas_deps

# %%
funding_info = (
    {
        "name": "Quick DECK GHG Forcing",
        "url": "No URL",
        "long_text": (
            "Financial support has been provided by the CMIP International Project Office (CMIP IPO), "
            "which is hosted by the European Space Agency (ESA), with staff provided by HE Space Operations Ltd."
        ),
    },
    {
        "name": "GHG Forcing For CMIP",
        "url": "climate.esa.int/supporting-modelling/cmip-forcing-ghg-concentrations/",
        "long_text": (
            "This research has been funded by the European Space Agency (ESA) as part of the "
            "GHG Forcing For CMIP project of the Climate Change Initiative (CCI) (ESA Contract No. 4000146681/24/I-LR-cl)."
        ),
    },
)

# %%
comment = (
    "Data compiled by Climate Resource, based on science by many others "
    "(see 'references*' attributes). "
    "For funding information, see the 'funding*' attributes."
)

# %%
non_input4mips_metadata_common = {
    "references": " --- ".join([v["reference"] for v in gas_deps]),
    "references_short_names": " --- ".join([v["short_name"] for v in gas_deps]),
    "references_dois": " --- ".join(
        [v["doi"] if ("doi" in v and v["doi"] is not None) else "No DOI" for v in gas_deps]
    ),
    "references_urls": " --- ".join([v["url"] for v in gas_deps]),
    "funding": " ".join([v["long_text"] for v in funding_info]),
    "funding_short_names": " --- ".join([v["name"] for v in funding_info]),
    "funding_urls": " --- ".join([v["url"] for v in funding_info]),
}
non_input4mips_metadata_common

# %% [markdown]
# ### Define variable renaming

# %%
gas_to_standard_name_renaming = {
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

# %%
gas_to_cmip_variable_renaming = {
    "co2": "co2",
    "ch4": "ch4",
    "n2o": "n2o",
    "c2f6": "c2f6",
    "c3f8": "c3f8",
    "c4f10": "c4f10",
    "c5f12": "c5f12",
    "c6f14": "c6f14",
    "c7f16": "c7f16",
    "c8f18": "c8f18",
    "cc4f8": "cc4f8",
    "ccl4": "ccl4",
    "cf4": "cf4",
    "cfc11": "cfc11",
    "cfc113": "cfc113",
    "cfc114": "cfc114",
    "cfc115": "cfc115",
    "cfc12": "cfc12",
    "ch2cl2": "ch2cl2",
    "ch3br": "ch3br",
    "ch3ccl3": "ch3ccl3",
    "ch3cl": "ch3cl",
    "chcl3": "chcl3",
    "halon1211": "halon1211",
    "halon1301": "halon1301",
    "halon2402": "halon2402",
    "hcfc141b": "hcfc141b",
    "hcfc142b": "hcfc142b",
    "hcfc22": "hcfc22",
    "hfc125": "hfc125",
    "hfc134a": "hfc134a",
    "hfc143a": "hfc143a",
    "hfc152a": "hfc152a",
    "hfc227ea": "hfc227ea",
    "hfc23": "hfc23",
    "hfc236fa": "hfc236fa",
    "hfc245fa": "hfc245fa",
    "hfc32": "hfc32",
    "hfc365mfc": "hfc365mfc",
    "hfc4310mee": "hfc4310mee",
    "nf3": "nf3",
    "sf6": "sf6",
    "so2f2": "so2f2",
    "cfc11eq": "cfc11eq",
    "cfc12eq": "cfc12eq",
    "hfc134aeq": "hfc134aeq",
}

# %% [markdown]
# ## Load CVs

# %%
raw_cvs_loader = get_raw_cvs_loader(config_step.input4mips_cvs_cv_source)
raw_cvs_loader

# %%
cvs = load_cvs_known_loader(raw_cvs_loader)
cvs.source_id_entries.source_ids

# %% [markdown]
# ### Write files

# %%
if config_step.start_year == 1:
    time_ranges_to_write = [
        range(1, 1000),
        range(1000, 1750),
        range(1750, int(global_mean_annual_data.time.dt.year[-1].values) + 1),
    ]

elif config_step.start_year == 1750:  # noqa: PLR2004
    time_ranges_to_write = [range(1750, int(global_mean_annual_data.time.dt.year[-1].values) + 1)]

else:
    raise NotImplementedError(config_step.start_year)

for start, end in itertools.pairwise(time_ranges_to_write):
    assert start[-1] == end[0] - 1

time_ranges_to_write

# %%
config_step.input4mips_out_dir.mkdir(exist_ok=True, parents=True)

time_dimension = "time"
for dat_resolution, grid_label, nominal_resolution, yearly_time_bounds in tqdman.tqdm(
    [
        (fifteen_degree_data, "gnz", "2500 km", False),
        # (half_degree_data, "05_deg_lat", False),
        (global_mean_monthly_data, "gm", "10000 km", False),
        (global_mean_annual_data, "gm", "10000 km", True),
        (hemispheric_mean_monthly_data, "gr1z", "10000 km", False),
        (hemispheric_mean_annual_data, "gr1z", "10000 km", True),
    ],
    desc="Resolutions",
):
    # TODO: calculate nominal resolution rather than guessing
    grid_info = " x ".join([f"{dat_resolution[v].size} ({v})" for v in dat_resolution.dims])
    print(f"Processing {grid_info} grid")

    variable_name_raw = str(dat_resolution.name)

    variable_name_output = gas_to_cmip_variable_renaming[variable_name_raw]
    ds_to_write = dat_resolution.to_dataset().rename_vars({variable_name_raw: variable_name_output})

    dimensions = tuple(str(v) for v in ds_to_write[variable_name_output].dims)
    print(f"{grid_label=}")
    print(f"{dimensions=}")

    # Use appropriate precision
    ds_to_write[variable_name_output] = ds_to_write[variable_name_output].astype(np.dtypes.Float32DType)
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

    metadata_minimum = Input4MIPsDatasetMetadataDataProducerMinimum(
        grid_label=grid_label,
        nominal_resolution=nominal_resolution,
        **metadata_minimum_common,
    )

    for time_range in time_ranges_to_write:
        ds_to_write_time_section = ds_to_write.sel(time=ds_to_write.time.dt.year.isin(time_range))

        input4mips_ds = Input4MIPsDataset.from_data_producer_minimum_information(
            data=ds_to_write_time_section,
            prepare_func=partial(
                prepare_ds_and_get_frequency,
                dimensions=dimensions,
                time_dimension=time_dimension,
                standard_and_or_long_names={
                    variable_name_output: {
                        "standard_name": gas_to_standard_name_renaming[variable_name_raw],
                        "long_name": variable_name_raw,
                    },
                },
                add_time_bounds=partial(
                    add_time_bounds,
                    monthly_time_bounds=not yearly_time_bounds,
                    yearly_time_bounds=yearly_time_bounds,
                ),
            ),
            metadata_minimum=metadata_minimum,
            cvs=cvs,
            dataset_category="GHGConcentrations",
            realm="atmos",
        )

        metadata_evolved = evolve(
            input4mips_ds.metadata,
            product="derived",
            comment=comment,
            doi=config.doi,
        )

        ds = input4mips_ds.data
        ds[variable_name_output].attrs["cell_methods"] = "area: time: mean"
        input4mips_ds = Input4MIPsDataset(
            data=ds,
            metadata=metadata_evolved,
            cvs=cvs,
            non_input4mips_metadata=non_input4mips_metadata_common,
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
#     --cv-source {config_step.input4mips_cvs_cv_source} \
#     --rglob-input "**/*{variable_name_output}*/**/*.nc"

# %%
config_step.input4mips_out_dir

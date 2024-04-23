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
import cf_xarray.units
import pint_xarray
import tqdm.autonotebook as tqdman
import xarray as xr
from input4mips_validation.dataset import Input4MIPsDataset
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
step_config_id: str = "ch4"  # config ID to select for this branch

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
fifteen_degree_data_raw = xr.load_dataarray(
    config_grid_crunching.fifteen_degree_monthly_file
).pint.quantify()
half_degree_data_raw = xr.load_dataarray(
    config_grid_crunching.half_degree_monthly_file
).pint.quantify()
gmnhsh_data_raw = xr.load_dataarray(
    config_grid_crunching.gmnhsh_mean_monthly_file
).pint.quantify()
gmnhsh_annual_data_raw = xr.load_dataarray(
    config_grid_crunching.gmnhsh_mean_annual_file
).pint.quantify()

# %% [markdown]
# ### Convert everything to a time axis

# %%
day = 15
fifteen_degree_data = local.xarray_time.convert_year_month_to_time(
    fifteen_degree_data_raw, day=day
)
half_degree_data = local.xarray_time.convert_year_month_to_time(
    half_degree_data_raw, day=day
)
gmnhsh_data = local.xarray_time.convert_year_month_to_time(gmnhsh_data_raw, day=day)
fifteen_degree_data

# %%
gmnhsh_annual_data = local.xarray_time.convert_year_to_time(
    gmnhsh_annual_data_raw, month=7, day=2, hour=12
)
gmnhsh_annual_data

# %% [markdown]
# ### Set common metadata

# %%
version = config.version

metadata_universal = dict(
    activity_id="input4MIPs",
    contact="zebedee.nicholls@climate-resource.com;malte.meinshausen@climate-resource.com",
    further_info_url="https://github.com/climate-resource/CMIP-GHG-Concentration-Generation",
    institution="Climate Resource, Fitzroy, Victoria 3065, Australia",
    institution_id="CR",
    mip_era="CMIP6Plus",
    target_mip="CMIP",
)

metadata_universal_optional: dict[str, str] = dict(
    # product="derived",
    # # TODO: check if there is a more exact grant agreement to refer to
    comment=(
        "[TBC which grant] Data produced by Climate Resource supported by funding "
        "from the CMIP IPO (Coupled Model Intercomparison Project International Project Office). "
        "This is an interim dataset, do not use in production."
    ),
    references="Meinshausen et al., 2017, GMD (https://doi.org/10.5194/gmd-10-2057-2017)",
)

# %% [markdown]
# ### Define variable renaming

# %%
gas_to_cmip_variable_renaming = {
    "co2": "mole_fraction_of_carbon_dioxide_in_air",
    "ch4": "mole_fraction_of_methane_in_air",
    "n2o": "mole_fraction_of_nitrous_oxide_in_air",
}

# %% [markdown]
# ### Write files

# %%
import input4mips_validation

# %%

# %%
time_dimension = "time"
for dat_resolution, yearly_time_bounds in tqdman.tqdm(
    [
        (fifteen_degree_data, False),
        (half_degree_data, False),
        (gmnhsh_data, False),
        (gmnhsh_annual_data, True),
    ],
    desc="Resolutions",
):
    variable_name_raw = dat_resolution.name
    variable_name_output = gas_to_cmip_variable_renaming[variable_name_raw]
    da_to_write = dat_resolution.to_dataset().rename_vars(
        {variable_name_raw: variable_name_output}
    )

    input4mips_ds = Input4MIPsDataset.from_raw_dataset(
        da_to_write,
        dimensions=da_to_write[variable_name_output].dims,
        time_dimension=time_dimension,
        metadata=metadata,
        metadata_optional=metadata_optional,
        add_time_bounds=partial(
            input4mips_validation.xarray_helpers.add_time_bounds,
            monthly_time_bounds=True,
        ),
    )
    written = input4mips_ds.write(root_data_dir=tmp_path)

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

    # input4mips_ds = Input4MIPsDataset.from_raw_dataset(
    #     dsv,
    #     dimensions=tuple(dsv.dims.keys()),
    #     time_dimension="time",
    #     metadata=metadata,
    #     metadata_optional=metadata_optional,
    #     add_time_bounds=partial(
    #         input4mips_validation.xarray_helpers.add_time_bounds,
    #         monthly_time_bounds=not yearly_time_bounds,
    #         yearly_time_bounds=yearly_time_bounds,
    #     ),
    # )

    # config_step.input4mips_out_dir.mkdir(exist_ok=True, parents=True)
    # print("Writing")
    # written = input4mips_ds.write(config_step.input4mips_out_dir)
    # print(f"Wrote: {written.relative_to(config_step.input4mips_out_dir)}")
    # print("")
    break

checklist_path = generate_directory_checklist(config_step.input4mips_out_dir)
# # Not sure if this is needed or not
# with open(config.input4mips_archive.complete_file_concentrations, "w") as fh:
#     fh.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

# %%

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
# # Write files for input4MIPs
#
# TODO:
#
# - process annual- and global-means in previous notebook, then use here
# - speak to Paul about infering metadata automatically, surely there already tools for this...
# - check correct grant reference with Eleanor

# %% [markdown]
# ## Imports

# %%
import cf_xarray  # noqa: F401 # required to add cf accessors
import cftime
import pint_xarray
import tqdm.autonotebook as tqdman
import xarray as xr
from carpet_concentrations.input4MIPs.dataset import (
    Input4MIPsDataset,
    Input4MIPsMetadata,
    Input4MIPsMetadataOptional,
)
from openscm_units import unit_registry

from local.config import load_config_from_file
from local.pydoit_nb.checklist import generate_directory_checklist
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "write_input4mips"

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
# ## Set-up unit registry

# %%
pint_xarray.accessors.default_registry = pint_xarray.setup_registry(unit_registry)

# %% [markdown]
# ## Load data
#
# In future, this should load all the gridded data, having already been crunched to global-means, annual-means etc.

# %%
assert False, "Load summarised and finer-grid data too"

# %%

# Put time back in and drop year so that writing still behaves in line with input4MIPs
# TODO: speak to Paul about whether this is sensible or not
tmp = gmnhsh_data_annual_mean.copy().rename({"year": "time"})
tmp = tmp.assign_coords(
    time=[cftime.DatetimeGregorian(y, 7, 2, 12) for y in tmp["time"]]
)
tmp

# %%
raw_gridded = xr.open_dataset(
    config_grid.processed_data_file, use_cftime=True
).pint.quantify()
raw_gridded

# %%
version = config.version

metadata_universal = dict(
    contact="zebedee.nicholls@climate-resource.com;malte.meinshausen@climate-resource.com",
    further_info_url="TBD TODO",
    institution="Climate Resource, Fitzroy, Victoria 3065, Australia",
    institution_id="CR",
    source_version=version,
    activity_id="input4MIPs",
    mip_era="CMIP6Plus",
    source=f"CR {version}",
)

metadata_universal_optional = dict(
    product="derived",
    # TODO: check if there is a more exact grant agreement to refer to
    comment=(
        "Data produced by Climate Resource supported by funding "
        "from the CMIP IPO (Coupled Model Intercomparison Project International Project Office)"
    ),
    references="[TODO]",
)

# %%
# TODO: use this pattern with rest of CV?
# wrap with pooch too?
import json
import urllib.request

cv_experiment_id_url = "https://raw.githubusercontent.com/WCRP-CMIP/CMIP6_CVs/master/CMIP6_experiment_id.json"

with urllib.request.urlopen(cv_experiment_id_url) as url:
    cv_experiment_id = json.load(url)

cv_experiment_id["version_metadata"]


def get_target_mip(scenario: str) -> str:
    target_mip = cv_experiment_id["experiment_id"][scenario]["activity_id"]
    assert len(target_mip) == 1
    target_mip = target_mip[0]

    return {"target_mip": target_mip}


# %%
import numpy as np


def lat_fifteen_deg(ds: xr.Dataset) -> bool:
    return np.allclose(
        ds.lat.values,
        np.array(
            [-82.5, -67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5]
        ),
    )


def get_dataset_category(variable: str) -> str:
    category_map = {
        "mole_fraction_of_carbon_dioxide_in_air": "GHGConcentrations",
        "mole_fraction_of_methane_in_air": "GHGConcentrations",
        "mole_fraction_of_nitrous_oxide_in_air": "GHGConcentrations",
    }

    try:
        return {"dataset_category": category_map[variable]}
    except KeyError:
        print(f"Missing {variable}")
        raise


def get_realm(variable: str) -> str:
    realm_map = {
        "mole_fraction_of_carbon_dioxide_in_air": "atmos",
        "mole_fraction_of_methane_in_air": "atmos",
        "mole_fraction_of_nitrous_oxide_in_air": "atmos",
    }

    try:
        return {"realm": realm_map[variable]}
    except KeyError:
        print(f"Missing {variable}")
        raise


def get_grid_label_nominal_resolution(ds: xr.Dataset) -> dict[str, str]:
    dims = ds.dims

    grid_label = None
    nominal_resolution = None
    grid = None
    if "lon" not in dims:
        if "lat" in dims:
            if lat_fifteen_deg(ds) and list(dims) == ["lat", "time"]:
                grid_label = "gn-15x360deg"
                nominal_resolution = "2500km"
                grid = "15x360 degree latitude x longitude"

        # elif "sector" in dims:
        #     # TODO: more stable handling of dims and whether bounds
        #     # have already been added or not
        #     if inp["sector"].size == 3 and list(sorted(dims)) == [
        #         "bounds",
        #         "sector",
        #         "time",
        #     ]:
        #         grid_label = "gr1-GMNHSH"
        #         nominal_resolution = "10000 km"

    if any([v is None for v in [grid_label, nominal_resolution]]):
        raise NotImplementedError(  # noqa: TRY003
            f"Could not determine grid_label for data: {ds}"
        )

    out = {
        "grid_label": grid_label,
        "nominal_resolution": nominal_resolution,
    }

    out_optional = {}
    if grid is not None:
        out_optional["grid"] = grid

    return out, out_optional


def get_frequency(ds: xr.Dataset) -> str:
    time_ax = ds["time"].values
    base_type = type(time_ax[0])
    # Horribly slow but ok for now
    monthly_steps = [
        base_type(y, m, 1, 0) for y, m in zip(ds["time"].dt.year, ds["time"].dt.month)
    ]

    if np.all(np.equal(time_ax, monthly_steps)):
        return {"frequency": "mon"}

    raise NotImplementedError(ds)


def infer_metadata_from_dataset(ds: xr.Dataset, scenario: str) -> dict[str, str]:
    if len(ds.data_vars) == 1:
        variable_id = list(ds.data_vars.keys())[0]
    else:
        raise AssertionError("Can only write one variable per file")  # noqa: TRY003

    # This seems wrong logic?
    source_id = scenario

    grid_info, grid_info_optional = get_grid_label_nominal_resolution(ds)

    out = {
        **grid_info,
        **get_target_mip(scenario),
        "source_id": source_id,
        # TODO: This seems like a misunderstanding too?
        "title": scenario,
        **get_dataset_category(variable_id),
        **get_realm(variable_id),
        # TODO: No idea what this is or means, is it meant to be
        # added as part of writing?
        "Conventions": "CF-1.6",
        **get_frequency(ds),
    }

    out_optional = {**grid_info_optional}

    return out, out_optional


# %%
rcmip_to_cmip_variable_renaming = {
    "Atmospheric Concentrations|CO2": "mole_fraction_of_carbon_dioxide_in_air",
    "Atmospheric Concentrations|CH4": "mole_fraction_of_methane_in_air",
    "Atmospheric Concentrations|N2O": "mole_fraction_of_nitrous_oxide_in_air",
}

# %%
for variable_id, dav in tqdman.tqdm(
    raw_gridded.loc[dict(region="World")].data_vars.items()
):
    dsv = dav.to_dataset().rename_vars(
        {variable_id: rcmip_to_cmip_variable_renaming[variable_id]}
    )

    scenario = list(np.unique(dsv["scenario"].values))
    assert len(scenario) == 1
    scenario = scenario[0]

    dsv = dsv.loc[{"scenario": scenario}]

    metadata_inferred, metadata_inferred_optional = infer_metadata_from_dataset(
        dsv, scenario
    )

    metadata = Input4MIPsMetadata(
        **metadata_universal,
        **metadata_inferred,
    )

    metadata_optional = Input4MIPsMetadataOptional(
        **metadata_universal_optional,
        **metadata_inferred_optional,
    )

    input4mips_ds = Input4MIPsDataset.from_metadata_autoadd_bounds_to_dimensions(
        dsv,
        dimensions=tuple(dsv.dims.keys()),
        metadata=metadata,
    )

    config_step.input4mips_out_dir.mkdir(exist_ok=True, parents=True)
    print("Writing")
    written = input4mips_ds.write(config_step.input4mips_out_dir)
    print(f"Wrote: {written.relative_to(config_step.input4mips_out_dir)}")
    print("")
    # break

checklist_path = generate_directory_checklist(config_step.input4mips_out_dir)
# # Not sure if this is needed or not
# with open(config.input4mips_archive.complete_file_concentrations, "w") as fh:
#     fh.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

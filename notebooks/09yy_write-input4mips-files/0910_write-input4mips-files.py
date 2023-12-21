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
# - process finer grid in previous notebook, then use here
# - speak to Paul about infering metadata automatically, surely there already tools for this...
# - check correct grant reference with Eleanor

# %% [markdown]
# ## Imports

# %%
import json
import urllib.request

import cf_xarray  # noqa: F401 # required to add cf accessors
import cftime
import numpy as np
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
config_gridded_data_processing = get_config_for_step_id(
    config=config, step="gridded_data_processing", step_config_id="only"
)

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
raw_gridded = xr.open_dataset(
    config_grid.processed_data_file, use_cftime=True
).pint.quantify()
raw_gridded

# %%
gmnhsh_data_global_hemisphere_mean = xr.open_dataset(
    config_gridded_data_processing.processed_data_file_global_hemispheric_means,
    use_cftime=True,
)
gmnhsh_data_global_hemisphere_mean

# %%
gmnhsh_data_annual_mean = xr.open_dataset(
    config_gridded_data_processing.processed_data_file_global_hemispheric_annual_means
)

# Put time back in and drop year so that writing still behaves in line with input4MIPs
# TODO: speak to Paul about whether this is sensible or not
gmnhsh_data_annual_mean = gmnhsh_data_annual_mean.rename({"year": "time"})
gmnhsh_data_annual_mean = gmnhsh_data_annual_mean.assign_coords(
    time=[
        cftime.DatetimeGregorian(y, 7, 2, 12) for y in gmnhsh_data_annual_mean["time"]
    ]
)
# TODO: No idea why this is needed
gmnhsh_data_annual_mean["time"].encoding = gmnhsh_data_global_hemisphere_mean[
    "time"
].encoding
gmnhsh_data_annual_mean

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
# input4MIPs CV here: https://github.com/PCMDI/input4MIPs-cmor-tables
# wrap with pooch too?

cv_experiment_id_url = "https://raw.githubusercontent.com/WCRP-CMIP/CMIP6_CVs/master/CMIP6_experiment_id.json"

if not cv_experiment_id_url.startswith("https:"):
    raise ValueError(cv_experiment_id_url)

with urllib.request.urlopen(cv_experiment_id_url) as url:  # noqa: S310 # checked above
    cv_experiment_id = json.load(url)

cv_experiment_id["version_metadata"]


def get_target_mip(experiment: str) -> dict[str, str]:
    """
    Get target MIP for a given experiment

    Parameters
    ----------
    experiment
        Experiment for which to find the parent MIP

    Returns
    -------
        Target MIP according to the controlled vocabulary
    """
    target_mip = cv_experiment_id["experiment_id"][experiment]["activity_id"]
    assert len(target_mip) == 1
    target_mip = target_mip[0]

    return {"target_mip": target_mip}


# %%


def lat_fifteen_deg(ds: xr.Dataset) -> bool:
    """
    Check if the latitude grid is our expected 15 degree grid

    Parameters
    ----------
    ds
        Dataset to check

    Returns
    -------
        Is the latitude grid our expected 15 degree grid?
    """
    return np.allclose(
        ds.lat.values,
        np.array(
            [-82.5, -67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5]
        ),
    )


def get_dataset_category(variable: str) -> dict[str, str]:
    """
    Get dataset category

    Parameters
    ----------
    variable
        Variable for which to retrieve the category

    Returns
    -------
        Dataset category metadata

    Raises
    ------
    KeyError
        We don't know the category for ``variable``
    """
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


def get_realm(variable: str) -> dict[str, str]:
    """
    Get realm

    Parameters
    ----------
    variable
        Variable for which to retrieve the realm

    Returns
    -------
        Realm of the variable metadata

    Raises
    ------
    KeyError
        We don't know the realm for ``variable``
    """
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


def get_grid_metadata(ds: xr.Dataset) -> tuple[dict[str, str], dict[str, str]]:
    """
    Get grid metadata for dataset

    Parameters
    ----------
    ds
        Dataset for which to derive the grid metadata

    Returns
    -------
        Compulsory metadata and optional metadata related to the grid

    Raises
    ------
    NotImplementedError
        We cannot determine the grid metadata for ``ds``
    """
    dims = ds.dims

    grid = None
    if "lon" not in dims:
        if "lat" in dims:
            if lat_fifteen_deg(ds) and list(dims) == ["lat", "time"]:
                # In CMIP6 input4MIPs, we used a grid label of "gn-15x360deg"
                # This doesn't seem to be in the CVs anymore
                # (https://github.com/PCMDI/input4MIPs-cmor-tables/blob/master/input4MIPs_grid_label.json)
                # so changing to "gr", which seems to be the best fit
                # TODO: discuss with Paul
                grid_label = "gr"
                nominal_resolution = "2500km"
                grid = "15x360 degree latitude x longitude"

        elif "sector" in dims:
            # In CMIP6 input4MIPs, we used a grid label of "gr1-GMNHSH"
            # This doesn't seem to be in the CVs anymore
            # (https://github.com/PCMDI/input4MIPs-cmor-tables/blob/master/input4MIPs_grid_label.json)
            # so changing to "gr1", which seems to be the best fit
            # TODO: discuss with Paul
            grid_label = "gr1"
            if "sector" in ds:
                hemispheric_means_lat_bounds = (
                    "0: -90.0, 90.0; 1: 0.0, 90.0; 2: -90.0, 0.0"
                )
                if ds["sector"].attrs["lat_bounds"] == hemispheric_means_lat_bounds:
                    nominal_resolution = "10000 km"
                    grid = "Global- and hemispheric-means"

    try:
        out = {
            "grid_label": grid_label,
            "nominal_resolution": nominal_resolution,
        }
    except NameError as exc:
        raise NotImplementedError(
            f"Could not determine grid_label for data: {ds}"
        ) from exc

    out_optional = {}
    if grid is not None:
        out_optional["grid"] = grid

    return out, out_optional


def get_frequency(ds: xr.Dataset) -> dict[str, str]:
    """
    Get frequency metadata of dataset

    Parameters
    ----------
    ds
        Dataset to get frequency metadata for

    Returns
    -------
        Frequency metadata

    Raises
    ------
    NotImplementedError
        We cannot infer the frequency metadata for ``ds``
    """
    time_ax = ds["time"].values
    base_type = type(time_ax[0])

    # Horribly slow but ok for now
    monthly_steps = [
        base_type(y, m, 1, 0) for y, m in zip(ds["time"].dt.year, ds["time"].dt.month)
    ]

    if np.all(np.equal(time_ax, monthly_steps)):
        return {"frequency": "mon"}

    # Horribly slow but ok for now
    yearly_steps = [base_type(y, 7, 2, 12) for y in ds["time"].dt.year]

    if np.all(np.equal(time_ax, yearly_steps)):
        return {"frequency": "yr"}

    raise NotImplementedError(ds)


def infer_metadata_from_dataset(
    ds: xr.Dataset, experiment: str
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Infer metadata from a dataset

    Parameters
    ----------
    ds
        Dataset for which to infer metdata

    experiment
        Scenario to which the data belongs (this data is not stored on the
        dataset at this stage so has to be passed in separately, we could of
        course change this pattern which might be smart)

    Returns
    -------
        Inferred metadata and inferred optional metadata

    Raises
    ------
    AssertionError
        There is more than one data variable in ``ds``
    """
    if len(ds.data_vars) == 1:
        # error mis-identified by ruff, think it's because it's xarray not a list
        variable_id = list(ds.data_vars.keys())[0]  # noqa: RUF015
    else:
        raise AssertionError("Can only write one variable per file")  # noqa: TRY003

    # This seems wrong logic?
    source_id = experiment

    grid_info, grid_info_optional = get_grid_metadata(ds)

    out = {
        **grid_info,
        **get_target_mip(experiment),
        "source_id": source_id,
        # TODO: This seems like a misunderstanding too?
        "title": experiment,
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
for dat_resolution, yearly_time_bounds in tqdman.tqdm(
    [
        (raw_gridded, False),
        (gmnhsh_data_global_hemisphere_mean, False),
        (gmnhsh_data_annual_mean, True),
    ],
    desc="Resolutions",
):
    for variable_id, dav in tqdman.tqdm(
        dat_resolution.loc[dict(region="World")].data_vars.items(),
        desc="Variables",
        leave=False,
    ):
        dsv = dav.to_dataset().rename_vars(
            {variable_id: rcmip_to_cmip_variable_renaming[variable_id]}
        )

        scenario_list = list(np.unique(dsv["scenario"].values))
        assert len(scenario_list) == 1
        scenario = scenario_list[0]

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
        if yearly_time_bounds:
            # TODO: remove this horrible hack and fix up the carpet_concentrations
            # behaviour instead
            ds_bnd = input4mips_ds.ds
            variable = "time"
            bname = f"{variable}_bounds"

            bounds_time = xr.DataArray(
                [
                    [cftime.datetime(y, 1, 1), cftime.datetime(y + 1, 1, 1)]
                    for y in ds_bnd["time"].dt.year
                ],
                dims=(variable, "bounds"),
                coords={variable: ds_bnd[variable], "bounds": [0, 1]},
            ).transpose(..., "bounds")

            ds_bnd.coords[bname] = bounds_time
            ds_bnd[variable].attrs["bounds"] = bname

            input4mips_ds = Input4MIPsDataset(ds=ds_bnd)

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

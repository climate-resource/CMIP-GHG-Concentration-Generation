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
# # Create gridded data
#
# - various bits of mucking around
# - `LatitudeSeasonalityGridder.calculate`
#     - this should be on a 15 degree latitude x monthly grid, everything else follows from that (Figure 1 from Meinshausen et al. 2017 https://gmd.copernicus.org/articles/10/2057/2017/gmd-10-2057-2017.pdf)
#         - 0.5 degree latitude x monthly grid is done with mean-preserving downscaling
#         - global- and hemispheric-monthly-means are latitudinal-weighted means
#         - global- and hemispheric-annual-means are latitudinal-weighted and time (month-weighted?) means
# - write to disk
#
# CSIRO notebook: https://github.com/climate-resource/csiro-hydrogen-esm-inputs/blob/main/notebooks/300_projected_concentrations/322_projection-gridding.py

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import cf_xarray  # noqa: F401 # required to add cf accessors
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint
import pint_xarray
import xarray as xr
from carpet_concentrations.gridders.latitude_seasonality_gridder import (
    LatitudeSeasonalityGridder,
)

# # TODO: remove from carpet_concentrations
# from carpet_concentrations.time import (
#     convert_time_to_year_month,
#     convert_year_month_to_time,
# )
from input4mips_validation.time import (
    convert_time_to_year_month,
    convert_year_month_to_time,
)
from openscm_units import unit_registry
from pydoit_nb.config_handling import get_config_for_step_id
from scmdata.run import BaseScmRun

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "grid"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% editable=true slideshow={"slide_type": ""}
config_quick_crunch = get_config_for_step_id(
    config=config, step="quick_crunch", step_config_id="only"
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Set-up unit registry

# %% editable=true slideshow={"slide_type": ""}
pint_xarray.accessors.default_registry = pint_xarray.setup_registry(unit_registry)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Load global-means

# %% editable=true slideshow={"slide_type": ""}
global_means = BaseScmRun(config_quick_crunch.processed_data_file_global_means)
global_means

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Convert to xarray

# %% editable=true slideshow={"slide_type": ""}
global_means_dict = {}
for vdf in global_means.groupby("variable"):
    variable = vdf.get_unique_meta("variable", True)
    gas = variable.split("Atmospheric Concentrations|")[-1]  # type: ignore
    units = vdf.get_unique_meta("unit", True)

    vxr = convert_time_to_year_month(
        vdf.to_xarray(dimensions=["region", "scenario"]).pint.quantify(  # type: ignore
            **{variable: units}
        )
    )

    global_means_dict[gas] = vxr

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load seasonality and latitudinal gradient
#
# To-do: pre-calculate and then load these rather than just creating them on the fly as is currently the case.

# %% editable=true slideshow={"slide_type": ""}
years = np.unique(global_means.time_points.years())
months = np.unique(global_means.time_points.months())
latitudes = np.arange(-82.5, 83, 15)

grid_shape = (*years.shape, *months.shape, *latitudes.shape)

# %% editable=true slideshow={"slide_type": ""}
gridders = {}
for gas, seasonality_amp, seasonality_shift, latitudinal_grad_m, units in (
    ("CO2", 2, -2, 1 / 82.5, "ppm"),
    ("CH4", 5, -1, 2 / 82.5, "ppb"),
    ("N2O", 0.01, -6, 0 / 82.5, "ppb"),
):
    if config.ci and gas != "CO2":
        # On CI, only crunch CO2
        continue

    seasonality_values = np.broadcast_to(
        seasonality_amp * np.sin(2 * np.pi * (months - seasonality_shift) / 12),
        grid_shape,
    )
    seasonality = xr.DataArray(
        seasonality_values,
        {"year": years, "lat": latitudes, "month": months},
        name="seasonality",
        attrs={"units": units},
    )

    latitudinal_gradient_values = np.broadcast_to(
        latitudinal_grad_m * latitudes, grid_shape
    )
    latitudinal_gradient = xr.DataArray(
        latitudinal_gradient_values,
        {"year": years, "month": months, "lat": latitudes},
        name="latitudinal_gradient",
        attrs={"units": units},
    )

    gridder = LatitudeSeasonalityGridder(
        xr.merge([seasonality, latitudinal_gradient])
        .cf.add_bounds("lat")
        .pint.quantify({"lat_bounds": "deg"})
    )

    gridders[gas] = gridder

# %% editable=true slideshow={"slide_type": ""}
gridders.keys()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Grid

# %% editable=true slideshow={"slide_type": ""}
gridded_concs_list = []
for gas, gridder in gridders.items():
    rcmip_variable = f"Atmospheric Concentrations|{gas}"

    res = gridder.calculate(global_means_dict[gas])
    # TODO: add cell_methods attribute?

    res_time_axis = convert_year_month_to_time(res).dropna(dim="time")
    gridded_concs_list.append(res_time_axis)

    ## Plotting
    display(res_time_axis)  # type: ignore[name-defined]  # noqa: F821

    print("Annual-, global-mean")
    res_time_axis.sel(region="World").groupby("time.year").mean().mean("lat")[
        rcmip_variable
    ].plot(hue="scenario")
    plt.show()

    print("Colour mesh plot")
    res_time_axis.sel(region="World")[rcmip_variable].plot.pcolormesh(
        x="time", y="lat", row="scenario", cmap="rocket_r", levels=100
    )
    plt.show()

    print("Contour plot fewer levels")
    res_time_axis.sel(region="World")[rcmip_variable].plot.contour(
        x="time", y="lat", row="scenario", cmap="rocket_r", levels=30
    )
    plt.show()

    print("Concs at different latitudes")
    res_time_axis.sel(region="World").sel(lat=[-87.5, 0, 87.5], method="nearest")[
        rcmip_variable
    ].plot.line(hue="lat", row="scenario", alpha=0.4)
    plt.show()

    print("Annual-mean concs at different latitudes")
    res_time_axis.sel(region="World").sel(
        lat=[-87.5, 0, 87.5], method="nearest"
    ).groupby("time.year").mean()[rcmip_variable].plot.line(
        hue="lat", row="scenario", alpha=0.4
    )
    plt.show()

    print("Flying carpet")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    tmp = res_time_axis.copy()
    tmp = tmp.assign_coords(time=tmp["time"].dt.year + tmp["time"].dt.month / 12)
    (
        tmp.sel(scenario="historical", region="World")[rcmip_variable]
        .isel(time=range(-150, 0))
        .plot.surface(
            x="time",
            y="lat",
            ax=ax,
            cmap="rocket_r",
            levels=30,
            # alpha=0.7,
        )
    )
    ax.view_init(15, -135, 0)  # type: ignore
    plt.tight_layout()
    plt.show()

    # break

## Join back together
gridded_concs = xr.merge(gridded_concs_list)
gridded_concs

# %% editable=true slideshow={"slide_type": ""}
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
gridded_concs.pint.dequantify().to_netcdf(config_step.processed_data_file)
config_step.processed_data_file

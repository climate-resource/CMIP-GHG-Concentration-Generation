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
# # Check output of adding satellite files

# %%
import re
from glob import glob

import matplotlib.pyplot as plt
import xarray as xr

# %%
file_list = glob(
    "/home/anna_lanteri/code/CMIP-GHG-Concentration-Generation/output-bundles/dev-test-run/data/raw/scaled_sat//200301_202312-C3S-L3_XCH4-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.6_*.nc"
)

prefix = "200301_202312-C3S-L3_XCH4-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.6_"

for file_path in file_list:
    filename = file_path.split("/")[-1]
    fit_name = re.search(rf"{re.escape(prefix)}?(.+)\.nc$", filename)
    print(fit_name.group(1))


# %% [raw]
# LINEAR_SEASONAL_LAT_FIT done
# NONLINEAR_LAT_STD_WEIGHT_FIT done
# LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT done
# LINEAR_STD_WEIGHT_FIT done
# LINEAR_SEASONAL_LAT_UNC_WEIGHT_FIT done
# LINEAR_SEASONAL_STD_WEIGHT_FIT done
# LINEAR_UNC_WEIGHT_FIT done
# LINEAR_FIT done
# LINEAR_SEASONAL_UNC_WEIGHT_FIT done
# NONLINEAR_LAT_UNC_WEIGHT_FIT done
# LINEAR_LAT_UNC_WEIGHT_FIT done
# NONLINEAR_LAT_FIT done
# LINEAR_LAT_FIT done
# LINEAR_LAT_STD_WEIGHT_FIT done
# LINEAR_SEASONAL_FIT done
#
# basic done

# %%
code_data_path = (
    "/output-bundles/dev-test-run/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-testing/"
)
code_path = "/home/anna_lanteri/code/CMIP-GHG-Concentration-Generation/"
date_path = "v20260729/"
output_path = f"{code_path}{code_data_path}/atmos/mon/co2/gnz/{date_path}"
file = "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212"
output_path

# %%
file_list = glob(output_path + "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-20221*.nc")

prefix = "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212_SAT_"

for file_path in file_list:
    filename = file_path.split("/")[-1]
    fit_name = re.search(rf"{re.escape(prefix)}?(.+)\.nc$", filename)
    print(fit_name.group(1))

# %% [markdown]
# # Comparisons

# %%
original_conc = xr.open_dataset(
    output_path + "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212.nc"
)
original_conc

# %%
linear_fit_conc = xr.open_dataset(
    output_path + "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212_SAT_LINEAR_FIT.nc"
)

# %%
original_conc.co2.plot()


# %%
def plot_comp_with_original(fit_name):
    fit_conc = xr.open_dataset(
        output_path
        + "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212_SAT_"
        + fit_name
        + ".nc"
    )
    (fit_conc.co2 - original_conc.co2).plot()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Comparing with original

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## No weights

# %%
plot_comp_with_original("LINEAR_FIT")

# %%
plot_comp_with_original("LINEAR_LAT_FIT")

# %%
plot_comp_with_original("LINEAR_SEASONAL_FIT")

# %%
plot_comp_with_original("LINEAR_SEASONAL_LAT_FIT")

# %%
plot_comp_with_original("NONLINEAR_LAT_FIT")

# %% [markdown]
# ## STD weight fits

# %%
plot_comp_with_original("LINEAR_STD_WEIGHT_FIT")

# %%
plot_comp_with_original("LINEAR_LAT_STD_WEIGHT_FIT")

# %%
plot_comp_with_original("LINEAR_SEASONAL_STD_WEIGHT_FIT")

# %%
plot_comp_with_original("LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT")

# %%
plot_comp_with_original("NONLINEAR_LAT_STD_WEIGHT_FIT")

# %% [markdown]
# ## UNC weights

# %%
plot_comp_with_original("LINEAR_UNC_WEIGHT_FIT")

# %%
plot_comp_with_original("LINEAR_LAT_UNC_WEIGHT_FIT")

# %%
plot_comp_with_original("LINEAR_SEASONAL_UNC_WEIGHT_FIT")

# %%
plot_comp_with_original("LINEAR_SEASONAL_LAT_UNC_WEIGHT_FIT")

# %%
plot_comp_with_original("NONLINEAR_LAT_UNC_WEIGHT_FIT")


# %% [markdown]
# # Comparing multiple together


# %%
def plot_comp_with_original(list_of_fits):
    fig, axes = plt.subplots(1, len(list_of_fits), figsize=(10, 5), sharey=True, sharex=True)

    # First pass: load all the diffs and figure out common vmin/vmax
    diffs = []
    for fit_name in list_of_fits:
        fit_conc = xr.open_dataset(
            output_path
            + "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212_SAT_"
            + fit_name
            + ".nc"
        )
        diffs.append(fit_conc.co2 - original_conc.co2)

    vmax = max(abs(d).max().item() for d in diffs)
    vmin = -vmax  # symmetric, good for diverging colormaps like RdBu_r

    # Second pass: plot with shared vmin/vmax, no individual colorbars
    for ax, fit_name, diff in zip(axes, list_of_fits, diffs):
        im = diff.plot(ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False, cmap="RdBu_r")
        ax.set_title(fit_name)

    plt.tight_layout()

    # Add one shared horizontal colorbar at the bottom, below all subplots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="CO2 difference")

    plt.show()


# %%
plot_comp_with_original(["LINEAR_FIT", "LINEAR_STD_WEIGHT_FIT"])

# %%
plot_comp_with_original(["LINEAR_SEASONAL_LAT_FIT", "LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT"])

# %%
plot_comp_with_original(["LINEAR_UNC_WEIGHT_FIT", "LINEAR_LAT_STD_WEIGHT_FIT"])

# %%
plot_comp_with_original(["LINEAR_FIT", "LINEAR_STD_WEIGHT_FIT", "LINEAR_UNC_WEIGHT_FIT"])

# %%
plot_comp_with_original(
    ["LINEAR_SEASONAL_LAT_FIT", "LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT", "LINEAR_SEASONAL_LAT_UNC_WEIGHT_FIT"]
)

# %%
plot_comp_with_original(["NONLINEAR_LAT_FIT", "NONLINEAR_LAT_STD_WEIGHT_FIT", "NONLINEAR_LAT_UNC_WEIGHT_FIT"])

# %%
plot_comp_with_original(
    ["LINEAR_UNC_WEIGHT_FIT", "LINEAR_LAT_UNC_WEIGHT_FIT", "LINEAR_SEASONAL_UNC_WEIGHT_FIT"]
)

# %%
plot_comp_with_original(
    ["LINEAR_UNC_WEIGHT_FIT", "LINEAR_LAT_UNC_WEIGHT_FIT", "LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT"]
)

# %%
plot_comp_with_original(
    [
        "LINEAR_UNC_WEIGHT_FIT",
        "LINEAR_LAT_UNC_WEIGHT_FIT",
        "LINEAR_SEASONAL_UNC_WEIGHT_FIT",
        "LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT",
        "NONLINEAR_LAT_UNC_WEIGHT_FIT",
    ]
)

# %%
LINEAR_FIT
LINEAR_LAT_FIT
LINEAR_SEASONAL_FIT
LINEAR_SEASONAL_LAT_FIT
NONLINEAR_LAT_FIT

LINEAR_STD_WEIGHT_FIT
LINEAR_LAT_STD_WEIGHT_FIT
LINEAR_SEASONAL_STD_WEIGHT_FIT
LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT
NONLINEAR_LAT_STD_WEIGHT_FIT

LINEAR_UNC_WEIGHT_FIT
LINEAR_LAT_UNC_WEIGHT_FIT
LINEAR_SEASONAL_UNC_WEIGHT_FIT
LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT
NONLINEAR_LAT_UNC_WEIGHT_FIT


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
original_nosat = xr.open_dataset(output_path + file + ".nc")
original_nosat

# %%
original_nosat.co2.plot()

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # Previous check

# %%
old_output_path = "/home/anna_lanteri/data/CMIP-GHG-Concentrations/"
# update this with your code execution date. Updates once a day
old_date = "v20260703/"
new_date = "v20260708/"
code_path = "/home/anna_lanteri/code/CMIP-GHG-Concentration-Generation/"
code_data_path = (
    "/output-bundles/dev-test-run/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-testing/"
)
new_output_path = f"{code_path}{code_data_path}/atmos/mon/co2/gnz/"
output_file = "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212.nc"

# %%
old_data = xr.open_dataset(old_output_path + output_file)
old_data

# %%
old_data.co2.plot()

# %%
new_old_data = xr.open_dataset(new_output_path + old_date + output_file)
new_old_data

# %%
new_new_data = xr.open_dataset(new_output_path + old_date + output_file)
new_new_data

# %%
new_old_data.co2.plot()

# %%
(new_new_data.co2 - new_old_data.co2).plot()

# %%
(new_new_data.co2 - old_data.co2).plot()

# %%

# %%

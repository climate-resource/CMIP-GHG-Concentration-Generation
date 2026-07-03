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

import xarray as xr

# %%
old_output_path = "/home/anna_lanteri/data/CMIP-GHG-Concentrations/"
new_output_path = "/home/anna_lanteri/code/CMIP-GHG-Concentration-Generation/output-bundles/dev-test-run/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-testing/atmos/mon/co2/gnz/v20260703/"
output_file = "co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_175001-202212.nc"

# %%
old_data = xr.open_dataset(old_output_path + output_file)
old_data

# %%
old_data.co2.plot()

# %%
new_data = xr.open_dataset(new_output_path + output_file)
new_data

# %%
new_data.co2.plot()

# %%
(new_data.co2 - old_data.co2).plot()

# %%

# %%

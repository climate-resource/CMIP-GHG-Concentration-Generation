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

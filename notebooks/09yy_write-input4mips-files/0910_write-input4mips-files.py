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
# - read already processed data off disk
# - set metadata that is universal
# - infer other metadata from data
#     - this is one to speak to Paul about, surely there already tools for this...
# - create complete set
# - write
#
# CSIRO notebook: https://github.com/climate-resource/csiro-hydrogen-esm-inputs/blob/main/notebooks/300_projected_concentrations/330_write-input4MIPs-files.py
#
# ```python
# from carpet_concentrations.input4MIPs.dataset import (
#     Input4MIPsDataset,
#     Input4MIPsMetadata,
#     Input4MIPsMetadataOptional,
# )
# ```

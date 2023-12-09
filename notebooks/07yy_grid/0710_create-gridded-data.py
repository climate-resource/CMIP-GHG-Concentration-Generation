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

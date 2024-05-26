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

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpsplines import MeanPreservingInterpolation as MPI

# %%
X = np.array(
    [
        2000,
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
    ]
)
X = X + 0.5
Y = np.array(
    [
        369.23319359,
        370.80045391,
        372.90803251,
        375.43797776,
        377.0998142,
        379.01836175,
        381.16035065,
        382.76322766,
        384.92829155,
        386.48951428,
        388.90205358,
        391.09214379,
        393.24470611,
        395.94703959,
        397.83299623,
        400.13632568,
        403.24974229,
        405.27395581,
        407.76253984,
        410.36763256,
        412.87306489,
        415.1473269,
        417.36471727,
    ]
)

x = np.arange(2000, 2023, 1 / 12)
x

# %%
mpi = MPI(yi=Y, xi=X, min_val=0.0)

# %%
plt.plot(
    X, Y, "ko", ms=4.5, mfc="none", mew=1.5, mec="k", label="averaged known process"
)
plt.plot(x, mpi(x), "r-", lw=1, label="mp-splines")

# %%
tmp = xr.load_dataarray(
    "/Users/znicholls/Documents/repos/CMIP-GHG-Concentration-Generation/output-bundles/dev-test-run/data/interim/co2/co2_global-annual-mean_allyears.nc"
)

# %%
X = tmp.year + 0.5
Y = tmp.values
x = np.arange(np.floor(np.min(X)) + 1 / 24, np.ceil(np.max(X)), 1 / 12)
x

# %%
mpi = MPI(yi=Y, xi=X, min_val=0.0)

# %%
plt.plot(
    X, Y, "ko", ms=4.5, mfc="none", mew=1.5, mec="k", label="averaged known process"
)
plt.plot(x, mpi(x), "rx", lw=1, label="mp-splines")
# plt.xlim([1, 30])
plt.xlim([1890, 1892])
plt.ylim([290, 300])
# plt.xlim([2000, 2025])

# %%
res = mpi(x)
for i in range(int(res.shape[0] / 12)):
    if not np.isclose(np.mean(res[i * 12 : (i + 1) * 12]), Y[i]):
        print(np.mean(res[i * 12 : (i + 1) * 12]))
        print(Y[i])
        print("help")
        break

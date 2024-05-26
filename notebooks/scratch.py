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
import scipy.interpolate

from local.mean_preserving_interpolation import mean_preserving_interpolation

# %%
MONTHS_PER_YEAR = 12
N_YEARS = 4
X = np.arange(MONTHS_PER_YEAR / 2 + 0.5, N_YEARS * MONTHS_PER_YEAR, MONTHS_PER_YEAR)
Y = X**2 + X + X**4
X

# %%
x = np.arange(1, N_YEARS * MONTHS_PER_YEAR + 1, 1)
x

# %%
plt.plot(X, Y)

# %%
plt.scatter(X, Y)
for df in np.arange(0.8, 1.11, 0.1):
    coefficients, intercept, knots, degree = mean_preserving_interpolation(
        X=X,
        Y=Y,
        x=x,
        degrees_freedom_scalar=df,
    )

    def interpolator(
        x,
    ):
        return (
            scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(x) + intercept
        )

    y = interpolator(x)
    plt.scatter(x, y, label=df)

plt.legend()

# %%

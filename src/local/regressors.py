"""
Regressors used during processing
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pint
import scipy.optimize
from attrs import define, field, validators

PredictCallableLike: TypeAlias = Callable[
    [pint.registry.UnitRegistry.Quantity], pint.registry.UnitRegistry.Quantity
]


@define
class WeightedQuantileRegressionResult:
    """
    Result of a weighted quantile regression
    """

    success: bool
    """
    Was the regression fitting successfully completed?
    """

    predict: PredictCallableLike | None
    """
    Function which can be used to predict y-values at given x-values, based on the regression
    """

    beta: pint.registry.UnitRegistry.Quantity | None
    """
    Beta vector that defines our regression result
    """


@define
class WeightedQuantileRegressor:
    """
    Regressor which performs a weighted quantile regression

    Add notes from here:
    https://docs.google.com/document/d/12r3B__DQGgwbfcI_BH6ZZjgEtOTVPN7h6c5NFhTLhJM/edit
    """

    quantile: float = field(default=0.5, validator=[validators.ge(0), validators.le(1)])
    """
    Quantile to fit
    """

    lambda_reg: float = field(default=1e-3, validator=validators.ge(0))
    """
    Lambda to use for regularisation
    """

    model_order: int = field(default=3, validator=validators.instance_of(int))
    """
    Order of the model to fit.

    1 means a linear model.
    """

    def fit(
        self,
        x: pint.registry.UnitRegistry.Quantity,
        y: pint.registry.UnitRegistry.Quantity,
        weights: pint.registry.UnitRegistry.Quantity,
    ) -> WeightedQuantileRegressionResult:
        """
        Fit a weighted quantile regression

        Parameters
        ----------
        x
            x-values

        y
            y-values

        weights
            Weights to apply to each value in ``x``/``y``

        Returns
        -------
            Result of the fit
        """
        if x.shape != y.shape:
            raise AssertionError

        if len(x.shape) > 1:
            raise AssertionError

        N = len(x)
        beta_len = self.model_order + 1

        c = self.lambda_reg * np.ones(2 * (beta_len + N))
        # # Can do something like this to prefer models with smaller higher-order terms.
        # for i in range(1, beta_len + 1):
        #     c[i] = self.lambda_reg ** (1 / (i + 1))
        #     c[beta_len + i] = self.lambda_reg ** (1 / (i + 1))

        c[2 * beta_len : 2 * beta_len + N] = weights * self.quantile
        c[2 * beta_len + N :] = weights * (1 - self.quantile)

        A = np.zeros((N, 2 * (beta_len + N)))
        A[
            (
                np.arange(N),
                np.arange(2 * beta_len, 2 * beta_len + N),
            )
        ] = 1
        A[
            (
                np.arange(N),
                np.arange(2 * beta_len + N, 2 * beta_len + 2 * N),
            )
        ] = -1

        for i in range(beta_len):
            A[:, i] = (x**i).m
            A[:, i + beta_len] = -(x**i).m

        b = y.m

        for maxiter in [1e5, 1e6, 1e7, 1e8, 1e9]:
            res = scipy.optimize.linprog(
                c,
                b_eq=b,
                A_eq=A,
                method="highs",
                bounds=(0, None),
                options=dict(maxiter=int(maxiter)),
            )
            if res.success:
                break

        else:
            print(f"didn't converge for {x=}")
            return WeightedQuantileRegressionResult(
                predict=None,
                beta=None,
                success=False,
            )

        Q = pint.get_application_registry().Quantity  # type: ignore
        beta_m = res.x[:beta_len] - res.x[beta_len : 2 * beta_len]
        # Ah yes, different units in one array aren't supported so this is super awkward to handle.
        beta = [Q(v, f"{y.units} / {(x[0] ** i).units}") for i, v in enumerate(beta_m)]

        def predict(
            x: pint.registry.UnitRegistry.Quantity,
        ) -> pint.registry.UnitRegistry.Quantity:
            """
            Predict a value based on the results of the regression

            Parameters
            ----------
            x
                x-point for which to predict the y-value

            Returns
            -------
                Predicted y-value, based on our regression
            """
            vec_prod = [beta[i] * (x**i) for i in range(beta_len)]

            return np.vstack(vec_prod).sum(axis=0)

        return WeightedQuantileRegressionResult(
            predict=predict,
            beta=beta,
            success=True,
        )

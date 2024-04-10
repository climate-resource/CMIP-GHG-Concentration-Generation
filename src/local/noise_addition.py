"""
Tools for adding noise to data
"""

from __future__ import annotations

import numpy as np
import pint
from attrs import define


@define
class NoiseAdderPercentageXNoise:
    """
    Adder of noise to data

    The x-axis noise increases as the x-value
    gets further away from some reference value.
    """

    x_ref: pint.UnitRegistry.Quantity
    """
    x-value to use as the reference.

    Data at this x-value is assumed to have no error in the x-axis.
    """

    x_relative_random_error: pint.UnitRegistry.Quantity
    """
    Rate at which the x-error grows as the data gets further from :attr:`x_ref`

    For example, if ``rel_x_error`` is 0.2, then a value which is 10 units
    away from :attr:`x_ref` will have an assumed x-error of 2.
    """

    y_random_error: pint.UnitRegistry.Quantity
    """
    Random error to apply to y-data
    """

    def add_noise(
        self, x: pint.UnitRegistry.Quantity, y: pint.UnitRegistry.Quantity
    ) -> tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity]:
        """
        Add noise to an input data set

        Parameters
        ----------
        x
            x-data to which to add noise

        y
            y-data to which to add noise

        Returns
        -------
            Tuple of ``(x_plus_noise, y_plus_noise)``
        """
        if x.shape != y.shape:
            msg = f"x and y must have the same shape. Received: {x.shape=}, {y.shape=}"
            raise ValueError(msg)

        # Uniform noise seems weird to me, but ok,
        # will copy Nicolai for now.
        # Different choice to Nicolai here, time error scales with age,
        # with zero being now rather than minimum value in input array.

        # TODO: add seed
        rng_x = np.random.default_rng(seed=None)
        x_plus_noise = x + (
            self.x_ref - x
        ) * self.x_relative_random_error * rng_x.uniform(
            low=-0.5, high=0.5, size=x.shape
        )

        rng_y = np.random.default_rng(seed=None)
        y_plus_noise = y + self.y_random_error * rng_y.uniform(
            low=-0.5, high=0.5, size=y.shape
        )

        out: tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity] = (
            x_plus_noise,
            y_plus_noise,
        )

        return out

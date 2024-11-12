"""
Tests of mean preserving interpolation
"""

from __future__ import annotations

import numpy as np
import pint

from local.mean_preserving_interpolation import mean_preserving_interpolation

# Test with uneven x-axis?
#    just raise NotImplementedError for now


# Test yearly interpolation
# Test latitudinal gradient interpolation
# The method we're using will mean we need a different bottom layer for each.
# That's fine, a general solution is super computationally expensive (as we have found out).
# Or just use Rymes-Myers for the spatial interpolation
def test_mean_preserving_interpolation():
    Q = pint.get_application_registry().Quantity

    y_start = Q([0, 0, 0, 1, 4, 9], "kg")
    # Put the +0.5 in the layer above the bottom layer for yearly interpolation
    # Test with different x spacing
    x_start = Q(np.arange(y_start.size) + 0.5, "yr")

    # Put the + 1 / 24 in the layer above the bottom layer for yearly interpolation
    x_out = Q(np.arange(0, y_start.size, 1 / 12) + 1 / 24, "yr")

    y_out = mean_preserving_interpolation(
        x_in=x_start,
        y_in=y_start,
        x_out=x_out,
    )
    breakpoint()

    assert False, "Add test of mean-preserving"
    assert False, "Add regression test of units and magnitude"


def test_mean_preserving_interpolation_unit_handling():
    # Should be able to use different units and get same answer (mod units)
    assert False

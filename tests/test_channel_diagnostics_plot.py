"""
Tests for choosing which column of a saved diagnostics table a figure should plot.
"""

import pandas as pd
import pytest

from scripts.analysis.make_channel_diagnostics_plot import (
    COMPARISON_COVERAGES,
    diagnostic_column,
)


def saved_table():
    """
    A table shaped like the one make_channel_diagnostics_plot writes
    """
    row = {"rate": 1.0, "expected_precision": 12.0, "coverage": 0.9}
    for coverage in COMPARISON_COVERAGES:
        row[f"min_precision_at_{coverage:g}"] = 10
        row[f"vocabulary_size_at_{coverage:g}"] = 100

    return pd.DataFrame([row])


def test_diagnostic_column_ignores_coverage_for_a_measure_that_does_not_depend_on_it():
    """
    Expected precision averages over everything the channel sends, so no vocabulary is involved
    """
    assert diagnostic_column(saved_table(), "expected_precision", 0.99) == (
        "expected_precision"
    )


def test_diagnostic_column_picks_the_column_for_the_coverage_asked_for():
    """
    Asking for a different coverage has to plot that coverage's numbers, not relabel the old ones.

    The table carries every coverage it measured, so redrawing at another one needs no channels and
    can be done anywhere. Getting this wrong would be silent: the axis would say 0.99 while the line
    showed 0.9.
    """
    df = saved_table()

    assert diagnostic_column(df, "min_precision", 0.99) == "min_precision_at_0.99"
    assert diagnostic_column(df, "vocabulary_size", 0.5) == "vocabulary_size_at_0.5"
    assert diagnostic_column(df, "min_precision", 0.999) == "min_precision_at_0.999"


def test_diagnostic_column_refuses_a_coverage_the_table_does_not_have():
    """
    A coverage that was never measured cannot be plotted, and the error should say what is available
    """
    with pytest.raises(ValueError, match="0.99"):
        diagnostic_column(saved_table(), "min_precision", 0.95)

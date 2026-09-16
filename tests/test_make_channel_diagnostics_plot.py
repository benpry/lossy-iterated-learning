"""
Tests for the channel diagnostics plotting script: the figures should show only the standard
uniform-source channels at a default coverage of 0.99, whatever else the sweep has cached.
"""

import struct

import pandas as pd
import pytest
from test_channel_analysis import TEST_DIMENSION, channel_filename, write_test_channel

from scripts.analysis.make_channel_diagnostics_plot import (
    COMPARISON_COVERAGES,
    Args,
    diagnose_one_channel,
    main,
    make_diagnostic_plot,
    reference_value,
    select_standard_channels,
)


def diagnostics_table(coverage: float = 0.99) -> pd.DataFrame:
    """
    A miniature of the saved diagnostics table, holding the standard channels alongside the
    decreasing-source and longer-lifespan variants the sweep also caches
    """
    rows = []
    for source, max_val in [("uniform", 21), ("uniform", 22), ("decreasing", 21)]:
        for rate, precision, vocabulary in [(0.0, 3.0, 1), (2.0, 30.0, 40)]:
            rows.append(
                {
                    "source_distribution": source,
                    "max_val": max_val,
                    "rate": rate,
                    "coverage": coverage,
                    "expected_precision": precision,
                    "min_precision": 3,
                    "vocabulary_size": vocabulary,
                    "source_precision": 33.0,
                    "source_min_precision": 3.0,
                    **{f"min_precision_at_{c:g}": 4 for c in COMPARISON_COVERAGES},
                    **{f"vocabulary_size_at_{c:g}": 50 for c in COMPARISON_COVERAGES},
                    "channel": f"{source}, max_val={max_val}",
                }
            )
    return pd.DataFrame(rows)


def test_default_coverage_is_099():
    assert Args().coverage == 0.99


def test_select_standard_channels_keeps_only_the_standard_variant():
    selected = select_standard_channels(diagnostics_table())

    assert set(selected["channel"]) == {"uniform, max_val=21"}
    assert len(selected) == 2


def test_select_standard_channels_fails_loudly_when_no_standard_channels_exist():
    variants_only = diagnostics_table().query("channel != 'uniform, max_val=21'")

    with pytest.raises(ValueError, match="uniform"):
        select_standard_channels(variants_only)


def test_reference_lines_compare_each_precision_panel_with_its_input_counterpart():
    df = select_standard_channels(diagnostics_table())

    # the average belief the channel sends against the average belief it receives
    assert reference_value(df, "expected_precision") == 33.0
    # the vaguest word it uses against the vaguest belief it can receive
    assert reference_value(df, "min_precision") == 3.0
    # the count of words has no input-side counterpart
    assert reference_value(df, "vocabulary_size") is None


def test_diagnose_one_channel_records_the_vaguest_possible_input(tmp_path):
    write_test_channel(tmp_path, beta=2.0)

    row = diagnose_one_channel(tmp_path / channel_filename(2.0), coverage=0.99)

    # pseudocounts start at one, so the vaguest belief a source can hold has one per parameter
    assert row["source_min_precision"] == TEST_DIMENSION


def test_make_diagnostic_plot_draws_the_column_for_the_coverage_asked_for():
    """
    Redrawing a table at another measured coverage has to plot that coverage's numbers; plotting
    the main column under a relabeled axis would be silent and wrong
    """
    df = select_standard_channels(diagnostics_table())  # measured at 0.99

    plot = make_diagnostic_plot(df, "min_precision", Args(coverage=0.5))

    assert plot.mapping["y"] == "min_precision_at_0.5"


def test_reference_value_fails_loudly_when_the_table_predates_a_reference_column():
    df = select_standard_channels(diagnostics_table()).drop(
        columns=["source_min_precision"]
    )

    with pytest.raises(ValueError, match="source_min_precision"):
        reference_value(df, "min_precision")


def test_a_table_predating_a_reference_column_still_plots_with_the_line_turned_off():
    df = select_standard_channels(diagnostics_table()).drop(
        columns=["source_min_precision"]
    )

    plot = make_diagnostic_plot(df, "min_precision", Args(reference_line=False))

    assert plot is not None


def test_main_can_save_the_figures_as_pngs(tmp_path):
    table_path = tmp_path / "channel_diagnostics.csv"
    diagnostics_table().to_csv(table_path, index=False)
    args = Args(
        output_file=table_path,
        figure_dir=tmp_path / "figures",
        reuse_table=True,
        figure_format="png",
    )

    main(args)

    for diagnostic in ["expected_precision", "min_precision", "vocabulary_size"]:
        assert (tmp_path / "figures" / f"channel-{diagnostic}_by_rate.png").exists()


def test_main_sizes_the_figures_as_asked(tmp_path):
    table_path = tmp_path / "channel_diagnostics.csv"
    diagnostics_table().to_csv(table_path, index=False)
    args = Args(
        output_file=table_path,
        figure_dir=tmp_path / "figures",
        reuse_table=True,
        figure_format="png",
        figure_width=3.0,
        figure_height=4.0,
    )

    main(args)

    # the width and height of a png, in pixels, straight from its header; plotnine renders
    # at 100 pixels per inch
    figure_path = tmp_path / "figures" / "channel-expected_precision_by_rate.png"
    width_px, height_px = struct.unpack(">II", figure_path.read_bytes()[16:24])
    assert (width_px, height_px) == (300, 400)


def test_precision_panels_start_at_zero():
    """
    The precision panels have a meaningful zero, and starting there keeps the size of the
    changes across rates honest
    """
    df = select_standard_channels(diagnostics_table())

    for diagnostic in ["expected_precision", "min_precision"]:
        figure = make_diagnostic_plot(df, diagnostic, Args()).draw()
        bottom, _ = figure.axes[0].get_ylim()
        assert bottom <= 0


def test_make_diagnostic_plot_uses_the_asked_for_font_size():
    df = select_standard_channels(diagnostics_table())

    plot = make_diagnostic_plot(df, "expected_precision", Args(font_size=30))

    assert plot.theme.themeables["text"].properties["size"] == 30


def test_main_plots_every_diagnostic_from_a_saved_table(tmp_path):
    table_path = tmp_path / "channel_diagnostics.csv"
    diagnostics_table().to_csv(table_path, index=False)
    args = Args(output_file=table_path, figure_dir=tmp_path / "figures", reuse_table=True)

    main(args)

    for diagnostic in ["expected_precision", "min_precision", "vocabulary_size"]:
        assert (tmp_path / "figures" / f"channel-{diagnostic}_by_rate.pdf").exists()


def test_main_refuses_a_saved_table_measured_at_a_different_coverage(tmp_path):
    table_path = tmp_path / "channel_diagnostics.csv"
    diagnostics_table(coverage=0.9).to_csv(table_path, index=False)
    args = Args(output_file=table_path, figure_dir=tmp_path / "figures", reuse_table=True)

    with pytest.raises(ValueError, match="coverage"):
        main(args)

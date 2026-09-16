"""Acceptance and numerical checks for the worked-example figure components."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pyprojroot import here


def test_beliefs_follow_the_two_heads_updates_and_the_stated_received_priors():
    from scripts.analysis.make_worked_example_plots import belief_table

    table = belief_table(0.9)
    assert list(zip(table["alpha"], table["beta"])) == [
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (2, 1),
        (3, 1),
    ]
    # Independent closed forms, including Alice's initial and updated beliefs.
    assert table["density_at_true_p"].tolist() == pytest.approx(
        [1, 1.8, 0.54, 0.972, 1.458, 1.9683, 1.8, 2.43]
    )
    assert table["log2_density_at_true_p"].tolist() == pytest.approx(
        np.log2(table["density_at_true_p"])
    )


def test_cli_exports_separate_transparent_components_and_auditable_values(tmp_path):
    output = tmp_path / "components"
    result = subprocess.run(
        [
            sys.executable,
            str(here("scripts/analysis/make_worked_example_plots.py")),
            "--output-dir",
            str(output),
        ],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    names = [
        "alice_prior",
        "alice_posterior",
        "bob_zero_prior",
        "bob_zero_posterior",
        "bob_intermediate_prior",
        "bob_intermediate_posterior",
        "bob_high_prior",
        "bob_high_posterior",
        "bob_scores",
        "alice",
        "bob",
        "transmission_zero",
        "transmission_intermediate",
        "transmission_high",
        "update_heads",
    ]
    for name in names:
        for extension in ("svg", "pdf", "png"):
            path = output / f"{name}.{extension}"
            assert path.stat().st_size > 0
        with Image.open(output / f"{name}.png") as image:
            expected_size = (600, 900) if name in ("alice", "bob") else (1200, 900)
            if name.startswith("transmission_"):
                expected_size = (1200, 360)
            elif name == "update_heads":
                expected_size = (600, 450)
            assert image.size == expected_size
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0))[3] == 0
        if name in names[:8]:
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            svg = ET.parse(output / f"{name}.svg", parser=parser)
            labels = {
                element.text.strip()
                for element in svg.iter()
                if element.tag is ET.Comment
            }
            assert labels == {"0", "1", "x", "p(x)"}
        elif name not in ("bob_scores", "update_heads"):
            svg = ET.parse(output / f"{name}.svg")
            assert not any(
                element.get("id", "").startswith("text_") for element in svg.iter()
            ), f"{name} should export without titles, axis labels, or tick labels"

    table = pd.read_csv(output / "beliefs.csv")
    assert table["component"].tolist() == names[:8]
    assert table["true_p"].tolist() == [0.9] * 8
    posteriors = table[table["component"].str.match("bob_.*_posterior")]
    assert posteriors["density_at_true_p"].tolist() == pytest.approx(
        [0.972, 1.9683, 2.43]
    )
    assert posteriors["log2_density_at_true_p"].tolist() == pytest.approx(
        np.log2([0.972, 1.9683, 2.43])
    )


def test_belief_panels_show_correct_densities_on_identical_axes():
    import matplotlib.pyplot as plt
    from scripts.analysis.make_worked_example_plots import (
        belief_table,
        make_belief_plot,
    )

    table = belief_table(0.9)
    # All distributions have integer parameters; check against their polynomial PDFs.
    polynomials = [
        lambda p: np.ones_like(p),
        lambda p: 2 * p,
        lambda p: 6 * p * (1 - p),
        lambda p: 12 * p**2 * (1 - p),
        lambda p: 20 * p**3 * (1 - p),
        lambda p: 30 * p**4 * (1 - p),
        lambda p: 2 * p,
        lambda p: 3 * p**2,
    ]
    for (_, row), pdf in zip(table.iterrows(), polynomials):
        figure = make_belief_plot(row)
        try:
            ax = figure.axes[0]
            curve, truth = ax.lines
            x, y = curve.get_data()
            np.testing.assert_allclose(y, pdf(x), atol=1e-12)
            assert x[0] == 0 and x[-1] == 1
            assert list(truth.get_xdata()) == [0.9, 0.9]
            assert ax.get_xlim() == (0, 1)
            assert ax.get_ylim() == (0, 3.2)
            assert ax.get_title() == ""
            assert ax.get_xlabel() == "x"
            assert ax.get_ylabel() == "p(x)"
            assert list(ax.get_xticks()) == [0, 1]
            assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1"]
            assert len(ax.get_yticks()) == 0
            assert [
                name for name, spine in ax.spines.items() if spine.get_visible()
            ] == ["bottom"]
            np.testing.assert_allclose(figure.get_size_inches(), [4, 3])
        finally:
            plt.close(figure)


def test_score_plot_uses_categorical_rates_and_base_two_log_densities():
    import matplotlib.pyplot as plt
    from scripts.analysis.make_worked_example_plots import belief_table, make_score_plot

    figure = make_score_plot(belief_table(0.9))
    try:
        ax = figure.axes[0]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "Zero",
            "Intermediate",
            "High",
        ]
        assert [bar.get_height() for bar in ax.patches] == pytest.approx(
            np.log2([0.972, 1.9683, 2.43])
        )
        assert [label.get_text() for label in ax.texts] == ["−0.04", "0.98", "1.28"]
        assert ax.get_ylim()[0] < -0.04
        assert "log₂" in ax.get_ylabel()
        assert "0.9" in ax.get_title()
    finally:
        plt.close(figure)


def test_people_are_distinct_unlabeled_vector_sketches():
    import matplotlib.pyplot as plt
    from scripts.analysis.make_worked_example_plots import make_person_plot

    sketches = []
    for person in ("alice", "bob"):
        figure = make_person_plot(person)
        try:
            ax = figure.axes[0]
            assert not ax.axison
            assert not ax.texts
            np.testing.assert_allclose(figure.get_size_inches(), [2, 3])
            assert len(ax.lines) >= 5  # Head, torso, arms, and legs are vector strokes.
            sketches.append([line.get_xydata().tolist() for line in ax.lines])
        finally:
            plt.close(figure)
    assert sketches[0] != sketches[1]
    with pytest.raises(ValueError, match="Unknown person"):
        make_person_plot("alcie")


def test_transmission_arrows_have_decreasing_disruption_and_consistent_direction():
    import matplotlib.pyplot as plt
    from scripts.analysis.make_worked_example_plots import make_transmission_plot

    deviations = []
    for rate in ("zero", "intermediate", "high"):
        figure = make_transmission_plot(rate)
        try:
            ax = figure.axes[0]
            assert not ax.axison
            assert not ax.texts
            np.testing.assert_allclose(figure.get_size_inches(), [4, 1.2])
            shaft, head = ax.lines
            x, y = shaft.get_data()
            assert x[0] < x[-1]
            assert y[0] == pytest.approx(0.5)
            assert y[-1] == pytest.approx(0.5)
            assert np.isnan(y).any() == (rate == "zero")
            deviations.append(np.nanmax(np.abs(y - 0.5)))
            head_x, head_y = head.get_data()
            assert head_x[1] == max(head_x)  # Arrowhead points to the right.
            assert (head_x[1], head_y[1]) == pytest.approx((x[-1], y[-1]))
        finally:
            plt.close(figure)
    assert deviations[0] > deviations[1] > deviations[2]
    assert deviations[2] == 0
    with pytest.raises(ValueError, match="Unknown channel rate"):
        make_transmission_plot("hgh")


def test_cli_rejects_invalid_truth_before_writing_any_figures(tmp_path):
    output = tmp_path / "components"
    result = subprocess.run(
        [
            sys.executable,
            str(here("scripts/analysis/make_worked_example_plots.py")),
            "--output-dir",
            str(output),
            "--true-p",
            "1.1",
        ],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "true_p must be finite and strictly between 0 and 1" in result.stderr
    assert not output.exists()


def test_evidence_update_symbol_shows_heads_above_a_rightward_arrow():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scripts.analysis.make_worked_example_plots import make_update_plot

    figure = make_update_plot()
    try:
        ax = figure.axes[0]
        assert not ax.axison
        np.testing.assert_allclose(figure.get_size_inches(), [2, 1.5])
        assert [text.get_text() for text in ax.texts] == ["H"]
        coin = next(patch for patch in ax.patches if isinstance(patch, Circle))
        assert ax.texts[0].get_position() == coin.center
        shaft, head = ax.lines
        x, y = shaft.get_data()
        assert x[0] < x[-1]
        assert max(y) < coin.center[1] - coin.radius
        head_x, head_y = head.get_data()
        assert head_x[1] == max(head_x)
        assert (head_x[1], head_y[1]) == (x[-1], y[-1])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("true_p", [0, 1, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_belief_table_rejects_truth_values_without_finite_interior_scores(true_p):
    from scripts.analysis.make_worked_example_plots import belief_table

    with pytest.raises(
        ValueError, match="true_p must be finite and strictly between 0 and 1"
    ):
        belief_table(true_p)

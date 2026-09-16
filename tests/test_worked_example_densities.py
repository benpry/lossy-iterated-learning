"""
Tests for the worked-example density script: for each channel rate in the paper's worked
example, Bob ends up with a different Beta posterior, and the script reports the density
of the true probability under each of those posteriors.
"""

import numpy as np
import pytest

from scripts.analysis.worked_example_densities import (
    Args,
    bob_posterior_densities,
    main,
)


def test_bob_posterior_densities_match_the_closed_form_beta_density():
    densities = bob_posterior_densities(true_p=0.7)

    assert list(densities["channel_rate"]) == ["zero", "intermediate", "high"]
    assert list(densities["posterior"]) == ["Beta(3, 2)", "Beta(5, 2)", "Beta(3, 1)"]
    assert densities["density"].tolist() == pytest.approx([1.764, 2.1609, 1.47])
    assert densities["log2_density"].tolist() == pytest.approx(
        np.log2([1.764, 2.1609, 1.47])
    )


def test_main_reports_the_density_of_the_true_probability_for_each_channel_rate(capsys):
    main(Args(true_p=0.7))

    output = capsys.readouterr().out
    # Bob's posteriors at zero, intermediate, and high channel rates, with the closed-form
    # Beta density at p=0.7: 12 * 0.7^2 * 0.3, 30 * 0.7^4 * 0.3, and 3 * 0.7^2.
    assert "Beta(3, 2)" in output and "1.7640" in output
    assert "Beta(5, 2)" in output and "2.1609" in output
    assert "Beta(3, 1)" in output and "1.4700" in output

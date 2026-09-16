"""
Tests for the worked example visualization script.
"""

import sys

from pyprojroot import here

sys.path.insert(0, str(here("scripts")))

from worked_example_visualization import (  # noqa: E402
    WORKED_EXAMPLE_BELIEFS,
    beta_belief_to_index,
    main,
)


def test_beta_belief_to_index():
    """
    Beliefs Beta(alpha, beta) map to grid indices where beta is the low-order digit
    """
    assert beta_belief_to_index(alpha=1, beta=1, max_val=5) == 0
    assert beta_belief_to_index(alpha=2, beta=1, max_val=5) == 5
    assert beta_belief_to_index(alpha=1, beta=2, max_val=5) == 1
    assert beta_belief_to_index(alpha=5, beta=5, max_val=5) == 24


def test_main_creates_components_and_channels_match_worked_example(tmp_path):
    """
    Integration test: running the script end-to-end saves one PDF per belief
    distribution and per channel, and the three channels reproduce the regimes
    described in the paper's worked example.
    """
    results = main(output_dir=tmp_path)

    for alpha, beta in WORKED_EXAMPLE_BELIEFS:
        assert (tmp_path / f"belief_beta_{alpha}_{beta}.pdf").exists()

    for regime in ["zero", "medium", "high"]:
        assert (tmp_path / f"channel_rate_{regime}.pdf").exists()
        assert (tmp_path / f"received_beliefs_rate_{regime}.pdf").exists()

    alice_idx = beta_belief_to_index(alpha=2, beta=1, max_val=5)

    # rate ~0: Bob receives the centroid belief Beta(2, 2) no matter what
    assert 0.0 <= results["zero"]["rate"] < 0.05
    centroid_idx = beta_belief_to_index(alpha=2, beta=2, max_val=5)
    assert results["zero"]["channel"][alice_idx, centroid_idx] > 0.99

    # medium rate: substantial probability of the prototype belief Beta(4, 2)
    prototype_idx = beta_belief_to_index(alpha=4, beta=2, max_val=5)
    assert results["medium"]["channel"][alice_idx, prototype_idx] > 0.2

    # high rate: Bob receives Alice's belief Beta(2, 1) with very high probability
    assert results["high"]["channel"][alice_idx, alice_idx] > 0.99

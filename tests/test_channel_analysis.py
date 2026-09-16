"""
Tests for the analysis of the outputs that channels use most often.
"""

import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.channel_analysis import (
    MARGINAL_TOLERANCE,
    ChannelSpec,
    analyze_channel_directory,
    analyze_channel_file,
    channel_cache_path,
    exact_beta_from_grid,
    fixed_point_residual,
    infer_encoding_max_val,
    marginal_consistency_gap,
    measure_remaining_movement,
    parse_channel_filename,
    resolve_channel_dir,
    self_transmission_probabilities,
    decode_pseudocounts,
    encode_pseudocounts,
    output_concentrations,
    summarize_marginal,
    summarize_precision,
    top_marginal_outputs,
)
from src.experiment import get_distortion_matrix
from src.info_theory import blahut_arimoto
from src.utils import uniform_source_density

TEST_DIMENSION = 2
TEST_MAX_VAL = 4
TEST_LIFESPAN = 1
TEST_ENCODING_MAX_VAL = TEST_MAX_VAL - TEST_LIFESPAN


def channel_filename(beta, source_distribution="uniform"):
    return (
        f"channel_dim-{TEST_DIMENSION}_max_val-{TEST_MAX_VAL}_beta-{beta}"
        f"_source-{source_distribution}_distortion-jeff_divergence.npy"
    )


def compute_test_channel(beta):
    """
    Compute a small channel to analyze.

    These tests are about reading a channel, not about how converged it is, so this caps the
    iterations rather than waiting for the tolerance the experiments use.
    """
    distortion_matrix = get_distortion_matrix(
        max_val=TEST_MAX_VAL,
        dimension=TEST_DIMENSION,
        lifespan=TEST_LIFESPAN,
        metric="jeff_divergence",
        cache=False,
    )
    source_distribution = jax.vmap(
        partial(
            uniform_source_density, dimension=TEST_DIMENSION, max_val=TEST_MAX_VAL
        )
    )(jnp.arange(TEST_MAX_VAL**TEST_DIMENSION))
    source_distribution = source_distribution / source_distribution.sum()

    return blahut_arimoto(
        source_distribution,
        distortion_matrix,
        beta,
        distortion_matrix.shape[1],
        max_iters=1000,
    )


def write_test_channel(directory, beta):
    """
    Cache a small channel in a directory, following the naming convention the experiment code uses
    """
    channel = compute_test_channel(beta)
    with open(directory / channel_filename(beta), "wb") as f:
        pickle.dump(channel, f)

    return channel


def test_analyze_channel_directory_end_to_end(tmp_path):
    """
    Analyzing a directory of cached channels reports each channel's most likely outputs
    """
    channel = write_test_channel(tmp_path, beta=2.0)
    channel_marginal = np.asarray(channel[1])

    df = analyze_channel_directory(tmp_path, top_k=3)

    assert len(df) == 3
    assert list(df["rank"]) == [1, 2, 3]

    # the channel-level metadata should come from the filename and the cached channel
    assert set(df["beta"]) == {2.0}
    assert set(df["source_distribution"]) == {"uniform"}
    assert set(df["dimension"]) == {TEST_DIMENSION}
    assert set(df["max_val"]) == {TEST_MAX_VAL}
    assert set(df["encoding_max_val"]) == {TEST_ENCODING_MAX_VAL}
    assert np.allclose(df["rate"], float(channel[2]))
    assert np.allclose(df["distortion"], float(channel[3]))
    assert (df["marginal_consistency_gap"] < MARGINAL_TOLERANCE).all()

    # the top outputs should be the highest-marginal-probability outputs, in order
    expected_indices = np.argsort(-channel_marginal, kind="stable")[:3]
    assert list(df["output_index"]) == list(expected_indices)
    assert np.allclose(df["marginal_prob"], channel_marginal[expected_indices])
    assert np.allclose(
        df["cumulative_prob"], np.cumsum(channel_marginal[expected_indices])
    )

    # the most likely output should be decoded into readable Dirichlet pseudocounts
    top_pseudocounts = df.iloc[0]["pseudocounts"]
    assert top_pseudocounts == "(2, 2)"


def test_marginal_consistency_gap_is_small_for_a_matching_marginal():
    """
    A channel and the marginal cached with it describe the same channel
    """
    channel, marginal = compute_test_channel(beta=2.0)[:2]
    spec = parse_channel_filename(channel_filename(beta=2.0))

    gap = marginal_consistency_gap(np.asarray(channel), np.asarray(marginal), spec)

    assert gap < MARGINAL_TOLERANCE


def test_marginal_consistency_gap_catches_a_mislabeled_source():
    """
    Reading a channel as if it had a different source distribution shows up as a large gap
    """
    channel, marginal = compute_test_channel(beta=2.0)[:2]
    mislabeled_spec = parse_channel_filename(
        channel_filename(beta=2.0, source_distribution="decreasing")
    )

    gap = marginal_consistency_gap(
        np.asarray(channel), np.asarray(marginal), mislabeled_spec
    )

    assert gap > MARGINAL_TOLERANCE


def test_analyze_channel_file_rejects_an_inconsistent_marginal(tmp_path):
    """
    A cached marginal that its channel does not induce is an error, not something to analyze
    """
    channel, marginal, *rest = compute_test_channel(beta=2.0)
    filepath = tmp_path / channel_filename(beta=2.0)
    with open(filepath, "wb") as f:
        pickle.dump((channel, jnp.roll(marginal, 1), *rest), f)

    with pytest.raises(ValueError, match="differs from the marginal"):
        analyze_channel_file(filepath, top_k=3)


def test_analyze_channel_directory_covers_every_channel(tmp_path):
    """
    Every cached channel in the directory gets analyzed
    """
    write_test_channel(tmp_path, beta=1.0)
    write_test_channel(tmp_path, beta=2.0)

    df = analyze_channel_directory(tmp_path, top_k=2)

    assert len(df) == 4
    assert sorted(set(df["beta"])) == [1.0, 2.0]


def test_parse_channel_filename():
    """
    Channel settings can be recovered from the cache filename
    """
    spec = parse_channel_filename(
        "channel_dim-3_max_val-21_beta-13.215_source-decreasing"
        "_distortion-jeff_divergence.npy"
    )

    assert spec == ChannelSpec(
        dimension=3,
        max_val=21,
        beta=13.215,
        source_distribution="decreasing",
        distortion_metric="jeff_divergence",
    )


def test_channel_cache_path_matches_the_experiment_naming():
    """
    The path of a cached channel can be rebuilt from its settings.

    This has to reproduce the name Experiment.compute_channel writes, including the way it rounds
    beta, so it relies on jax running with x64 enabled the way the experiments do.
    """
    spec = ChannelSpec(
        dimension=3,
        max_val=21,
        beta=0.171,
        source_distribution="uniform",
        distortion_metric="jeff_divergence",
    )

    path = channel_cache_path(spec, "/scr/benpry/cache/channels")

    assert path.parent == Path("/scr/benpry/cache/channels")
    assert path.name == (
        "channel_dim-3_max_val-21_beta-0.171_source-uniform_distortion-jeff_divergence.npy"
    )


def test_channel_cache_path_round_trips_through_parsing():
    """
    Building a cache path and parsing it again gives back the settings it was built from
    """
    spec = ChannelSpec(
        dimension=3,
        max_val=22,
        beta=13.215,
        source_distribution="decreasing",
        distortion_metric="jeff_divergence",
    )

    assert parse_channel_filename(channel_cache_path(spec, "/scr/benpry/cache/channels")) == spec


def test_parse_channel_filename_rejects_unexpected_names():
    """
    A file that does not follow the channel naming convention is an error, not a silent skip
    """
    with pytest.raises(ValueError, match="does not look like a cached channel"):
        parse_channel_filename("some_other_file.npy")


def test_infer_encoding_max_val():
    """
    The maximum encoded pseudocount is recovered from the number of channel outputs
    """
    assert infer_encoding_max_val(n_encodings=8000, dimension=3) == 20
    assert infer_encoding_max_val(n_encodings=9, dimension=2) == 3


def test_infer_encoding_max_val_rejects_inconsistent_shapes():
    """
    A number of outputs that is not a perfect power of the dimension is an error
    """
    with pytest.raises(ValueError, match="not a perfect"):
        infer_encoding_max_val(n_encodings=8001, dimension=3)


def test_top_marginal_outputs():
    """
    The top outputs are ranked by marginal probability and decoded into pseudocounts
    """
    # indices 0, 1, 2, 3 correspond to pseudocounts (1, 1), (2, 1), (3, 1), (1, 2)
    marginal = np.array([0.1, 0.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    df = top_marginal_outputs(marginal, dimension=2, encoding_max_val=3, top_k=2)

    assert list(df["rank"]) == [1, 2]
    assert list(df["output_index"]) == [2, 3]
    assert list(df["pseudocounts"]) == ["(3, 1)", "(1, 2)"]
    assert np.allclose(df["marginal_prob"], [0.6, 0.3])
    assert np.allclose(df["cumulative_prob"], [0.6, 0.9])
    assert np.allclose(df["concentration"], [4, 3])
    assert list(df["mean_probs"]) == ["(0.750, 0.250)", "(0.333, 0.667)"]


def test_top_marginal_outputs_caps_at_number_of_outputs():
    """
    Asking for more outputs than the channel has returns every output
    """
    marginal = np.array([0.5, 0.25, 0.25, 0.0])

    df = top_marginal_outputs(marginal, dimension=2, encoding_max_val=2, top_k=10)

    assert len(df) == 4


def test_summarize_marginal():
    """
    The marginal is summarized by its entropy and the size of its effective support
    """
    marginal = np.array([0.5, 0.25, 0.25, 0.0])

    summary = summarize_marginal(marginal)

    assert np.isclose(summary["marginal_entropy_bits"], 1.5)
    assert np.isclose(summary["marginal_perplexity"], 2**1.5)
    assert summary["n_outputs_90_pct_mass"] == 3
    assert summary["n_outputs_used"] == 3


def test_fixed_point_residual_is_tiny_for_a_converged_channel():
    """
    A converged channel is one the Blahut-Arimoto update would leave alone
    """
    distortion_matrix = get_distortion_matrix(
        max_val=TEST_MAX_VAL,
        dimension=TEST_DIMENSION,
        lifespan=TEST_LIFESPAN,
        metric="jeff_divergence",
        cache=False,
    )
    source_p = jnp.ones(distortion_matrix.shape[0]) / distortion_matrix.shape[0]
    channel = blahut_arimoto(
        source_p, distortion_matrix, 2.0, distortion_matrix.shape[1], tolerance=1e-12
    )[0]

    residual = fixed_point_residual(
        np.asarray(channel), np.asarray(source_p), np.asarray(distortion_matrix), beta=2.0
    )

    assert residual < 1e-10


def test_fixed_point_residual_is_large_for_a_channel_that_stopped_early():
    """
    A channel that stopped short of the fixed point is visibly not at it
    """
    distortion_matrix = get_distortion_matrix(
        max_val=TEST_MAX_VAL,
        dimension=TEST_DIMENSION,
        lifespan=TEST_LIFESPAN,
        metric="jeff_divergence",
        cache=False,
    )
    source_p = jnp.ones(distortion_matrix.shape[0]) / distortion_matrix.shape[0]
    channel = blahut_arimoto(
        source_p, distortion_matrix, 2.0, distortion_matrix.shape[1], max_iters=5
    )[0]

    residual = fixed_point_residual(
        np.asarray(channel), np.asarray(source_p), np.asarray(distortion_matrix), beta=2.0
    )

    assert residual > 1e-6


def test_exact_beta_from_grid_recovers_full_precision():
    """
    A beta read back from a cache filename is matched to the grid value it was rounded from.

    Channels are cached under a beta rounded to three decimals, which is not precise enough to
    rebuild the channel's own distortion kernel, so the full precision value has to come back from
    the grid the sweep used.
    """
    betas = np.logspace(-4, 10, num=30, base=2.0)

    # every beta on the grid comes back from the value its cache filename would carry
    for index in range(len(betas)):
        rounded = parse_channel_filename(
            channel_cache_path(
                ChannelSpec(3, 21, float(betas[index]), "uniform", "jeff_divergence"),
                "/scr/benpry/cache/channels",
            )
        ).beta
        assert exact_beta_from_grid(rounded, betas) == betas[index]


def test_exact_beta_from_grid_rejects_a_beta_that_is_not_on_the_grid():
    """
    A cached channel from some other sweep is an error rather than a silent mismatch
    """
    betas = np.logspace(-4, 10, num=30, base=2.0)

    with pytest.raises(ValueError, match="does not match any beta"):
        exact_beta_from_grid(0.5, betas)


# the beta where this small problem's codebook changes shape, and convergence slows to a crawl
CREEPING_BETA = 1.355
# enough iterations to be well into the geometric tail, but well short of converged
STOPPED_SHORT_ITERS = 800_000
REFERENCE_ITERS = 20_000_000


def creeping_problem():
    """
    A problem and beta where the channel creeps: every step is minute, the distance left is not
    """
    distortion_matrix = np.asarray(
        get_distortion_matrix(
            max_val=5, dimension=2, lifespan=1, metric="jeff_divergence", cache=False
        )
    )
    source_p = np.ones(distortion_matrix.shape[0]) / distortion_matrix.shape[0]

    return source_p, distortion_matrix


def test_measure_remaining_movement_matches_the_distance_actually_left():
    """
    What it reports is the distance the channel really does still have to travel.

    A channel stopped near the beta where the codebook changes shape moves in steps hundreds of
    thousands of times smaller than the distance remaining, so the distance cannot be read off a
    step and has to be inferred from how fast the steps are decaying. Measuring that decay is the
    whole job, and it is why this iterates across windows rather than comparing two steps.
    """
    source_p, distortion_matrix = creeping_problem()

    stopped_short = np.asarray(
        blahut_arimoto(
            source_p,
            distortion_matrix,
            CREEPING_BETA,
            distortion_matrix.shape[1],
            max_iters=STOPPED_SHORT_ITERS,
            tolerance=0.0,
        )[0]
    )
    settled = np.asarray(
        blahut_arimoto(
            source_p,
            distortion_matrix,
            CREEPING_BETA,
            distortion_matrix.shape[1],
            max_iters=REFERENCE_ITERS,
            tolerance=0.0,
        )[0]
    )
    distance_left = float(np.abs(stopped_short - settled).sum(axis=1).max() / 2)

    measured = measure_remaining_movement(
        stopped_short, source_p, distortion_matrix, CREEPING_BETA
    )

    # the channel is genuinely still far from done, so there is something to get right
    assert distance_left > 1e-7
    assert measured["remaining_movement"] == pytest.approx(distance_left, rel=0.25)


def test_measure_remaining_movement_is_zero_for_a_channel_at_the_fixed_point():
    """
    A channel that has stopped moving has nothing left to travel, however slowly it got there
    """
    source_p, distortion_matrix = creeping_problem()
    settled = np.asarray(
        blahut_arimoto(
            source_p,
            distortion_matrix,
            16.0,
            distortion_matrix.shape[1],
            max_iters=REFERENCE_ITERS,
            tolerance=0.0,
        )[0]
    )

    measured = measure_remaining_movement(settled, source_p, distortion_matrix, 16.0)

    assert measured["remaining_movement"] == 0.0


def test_output_concentrations_sums_the_pseudocounts_of_every_output():
    """
    Each output's concentration is the sum of the pseudocounts its index decodes to
    """
    # with two parameters each running 1..3, index 0 is (1, 1) and index 8 is (3, 3)
    concentrations = output_concentrations(dimension=2, encoding_max_val=3)

    assert concentrations.tolist() == [2, 3, 4, 3, 4, 5, 4, 5, 6]
    # and it agrees with decoding the indices one at a time
    for index in range(9):
        expected = decode_pseudocounts(index, 2, 3).sum()
        assert concentrations[index] == expected


def test_expected_precision_averages_concentration_under_the_marginal():
    """
    Expected precision is how confident a belief the channel sends on average
    """
    # all the mass on index 0, which is (1, 1)
    point_mass = np.zeros(9)
    point_mass[0] = 1.0
    assert summarize_precision(point_mass, 2, 3)["expected_precision"] == pytest.approx(2.0)
    assert summarize_precision(point_mass, 2, 3)["precision_sd"] == pytest.approx(0.0)

    # spread evenly over every output, so the mean of [2,3,4,3,4,5,4,5,6]
    uniform = np.ones(9) / 9
    assert summarize_precision(uniform, 2, 3)["expected_precision"] == pytest.approx(4.0)
    assert summarize_precision(uniform, 2, 3)["precision_sd"] == pytest.approx(
        np.std([2, 3, 4, 3, 4, 5, 4, 5, 6])
    )


def test_expected_precision_rises_as_the_channel_sends_more_confident_beliefs():
    """
    A channel that only ever sends (3, 3) is more precise than one that only sends (1, 1)
    """
    vague, confident = np.zeros(9), np.zeros(9)
    vague[0] = 1.0
    confident[8] = 1.0

    assert (
        summarize_precision(confident, 2, 3)["expected_precision"]
        > summarize_precision(vague, 2, 3)["expected_precision"]
    )


def test_summarize_precision_reports_the_vocabulary_and_its_least_confident_member():
    """
    The vocabulary is the set of words that carries most of what the channel sends
    """
    # mass only on index 0 = (1, 1) and index 8 = (3, 3), concentrations 2 and 6
    two_words = np.zeros(9)
    two_words[[0, 8]] = 0.5

    summary = summarize_precision(two_words, 2, 3)
    assert summary["vocabulary_size"] == 2
    assert summary["min_precision"] == 2
    assert summary["max_precision"] == 6
    assert summary["expected_precision"] == pytest.approx(4.0)

    one_word = np.zeros(9)
    one_word[8] = 1.0
    summary = summarize_precision(one_word, 2, 3)
    assert summary["vocabulary_size"] == 1
    assert summary["min_precision"] == 6


def test_summarize_precision_ignores_a_word_the_channel_almost_never_sends():
    """
    A vague word sent once in a thousand transmissions should not be reported as the vaguest word
    the channel uses.

    This is the whole reason the vocabulary is defined by the mass it covers rather than by a
    probability floor. Under a floor of 1e-10 the rare word below counts, and the channel looks as
    though it can express near-total ignorance when in practice it never does.
    """
    # 99.9% of the mass on (3, 3) with concentration 6, and a thousandth on (1, 1) with 2
    marginal = np.zeros(9)
    marginal[8] = 0.999
    marginal[0] = 0.001

    summary = summarize_precision(marginal, 2, 3, coverage=0.9)
    assert summary["vocabulary_size"] == 1
    assert summary["min_precision"] == 6

    # asking for essentially all of the mass does pick the rare word up
    summary = summarize_precision(marginal, 2, 3, coverage=0.9999)
    assert summary["vocabulary_size"] == 2
    assert summary["min_precision"] == 2


def test_summarize_precision_keeps_a_vocabulary_however_thinly_the_mass_is_spread():
    """
    A channel that spreads its mass evenly over many words still has a vocabulary.

    A fixed probability floor breaks here: once the mass is spread over enough outputs, every one of
    them falls below any fixed threshold and the vocabulary looks empty.
    """
    spread_thin = np.ones(9) / 9

    summary = summarize_precision(spread_thin, 2, 3, coverage=0.9)
    assert summary["vocabulary_size"] == 9
    assert summary["min_precision"] == 2
    assert summary["max_precision"] == 6


def test_summarize_precision_refuses_a_marginal_that_is_not_a_distribution():
    """
    A marginal that carries no probability has no vocabulary, and saying so beats returning a NaN
    """
    with pytest.raises(ValueError, match="no probability"):
        summarize_precision(np.zeros(9), 2, 3)


def test_encode_pseudocounts_inverts_decode_pseudocounts():
    """
    Finding whether a belief survives transmission means locating it in the output space
    """
    for index in range(3**2):
        pseudocounts = decode_pseudocounts(index, dimension=2, encoding_max_val=3)
        assert encode_pseudocounts(pseudocounts, encoding_max_val=3) == index


def test_self_transmission_probabilities_reads_the_diagonal_across_two_index_spaces():
    """
    The probability a belief comes back as itself, for the beliefs that can come back at all.

    Inputs range over more pseudocount values than outputs do, because agents observe more over a
    lifetime than they can say, so the "diagonal" is not the diagonal of the channel matrix. Inputs
    with a pseudocount above what the outputs can express are left out: there is no cell for them.
    """
    # inputs are the 9 vectors over 1..3, outputs the 4 over 1..2
    channel = np.zeros((9, 4))
    # (1, 1) is input 0 and output 0; (2, 1) is input 1 and output 1
    channel[0, 0] = 0.7
    channel[1, 1] = 0.2
    # (1, 2) is input 3 under max_val 3 but output 2 under encoding_max_val 2
    channel[3, 2] = 0.9
    # (2, 2) is input 4 and output 3, and never comes back
    channel[4, 3] = 0.0

    indices, probabilities = self_transmission_probabilities(
        channel, dimension=2, max_val=3, encoding_max_val=2
    )

    # only the four inputs whose pseudocounts all fit inside 1..2 are representable
    assert indices.tolist() == [0, 1, 3, 4]
    assert probabilities.tolist() == pytest.approx([0.7, 0.2, 0.9, 0.0])


def test_self_transmission_probabilities_covers_every_representable_input():
    """
    A channel whose outputs span the same range as its inputs has every input on the diagonal
    """
    channel = np.eye(4)

    indices, probabilities = self_transmission_probabilities(
        channel, dimension=2, max_val=2, encoding_max_val=2
    )

    assert indices.tolist() == [0, 1, 2, 3]
    assert probabilities.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_resolve_channel_dir_prefers_an_explicit_directory(monkeypatch):
    """
    A directory given on the command line is used whatever the environment says
    """
    monkeypatch.setenv("SCR_ROOT_DIR", "/scr/somebody")

    assert resolve_channel_dir(Path("/somewhere/else")) == Path("/somewhere/else")


def test_resolve_channel_dir_falls_back_to_the_cache_on_the_cluster(monkeypatch):
    """
    With nothing given, the channels are wherever the cluster keeps them
    """
    monkeypatch.setenv("SCR_ROOT_DIR", "/scr/somebody")

    assert resolve_channel_dir(None) == Path("/scr/somebody/cache/channels")


def test_resolve_channel_dir_says_what_to_do_when_there_is_no_cluster_cache(monkeypatch):
    """
    Off the cluster there is no SCR_ROOT_DIR, and the error has to say what to do about it.

    The analysis scripts are run on laptops to redraw figures from a saved table, where reading a
    channel is neither possible nor needed, so this must not be a bare KeyError from deep inside
    argument parsing.
    """
    monkeypatch.delenv("SCR_ROOT_DIR", raising=False)

    with pytest.raises(ValueError, match="--channel-dir"):
        resolve_channel_dir(None)

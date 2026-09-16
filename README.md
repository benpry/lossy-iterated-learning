# lossy-iterated-learning
Code and data for the paper "Lossy communication constrains cumulative cultural evolution" by Prystawski, Arumugam, and Goodman.

## Setup

To reproduce our results, you should first create a conda environment and install the dependencies in `requirements.txt`. Next, you should install the source code for this package.

```bash
conda create -n rd-culture python=3.12 pip
conda activate rd-culture
pip install -r requirements.txt
pip install -e .
```

The simulations cache large intermediate results (distortion matrices and channels) on local disk. Set the `SCR_ROOT_DIR` environment variable to a directory with plenty of space; caches are written under `$SCR_ROOT_DIR/cache/`. Only the simulation and channel-analysis scripts need it. The plotting scripts read the tables in `data/` and run anywhere.

## Code layout

The `src/` directory contains the core code and utilities for running simulations.

- `experiment.py`: the `Experiment` class, which computes (or loads from the cache) the channel for each rate limit and runs the analytic iterated learning simulation described in Appendix C of the paper.
- `info_theory.py`: information-theoretic utilities, including our implementation of the Blahut-Arimoto algorithm. The algorithm iterates until its estimate of the movement still to come falls below a tolerance, rather than for a fixed number of steps, and returns that estimate alongside the channel so that callers can refuse a channel that did not converge.
- `utils.py`: utilities for converting between pseudocount vectors and indices, computing divergences between Dirichlet distributions, and building the observation transition matrix.
- `channel_analysis.py`: code for reading cached channels and measuring what they do: which outputs they use, how confident those outputs are, how often a belief is transmitted as itself, and how far a cached channel is from the Blahut-Arimoto fixed point.
- `process_results.py`: computes metrics like expected scores from raw vectors of belief proportions.

## Running the simulations

Each configuration file in `configs/` describes one model variant: the true probabilities, the maximum pseudocount, the number of observations per generation, the source distribution, and how tightly the channels have to converge (`convergence_tolerance`, in total variation per row of the channel, and `max_channel_iters`).

Computing the channels is by far the most expensive step, so it is done first and separately:

1. `scripts/compute_channels.py --config <name>` computes the channel for every rate limit in a config and writes them to `$SCR_ROOT_DIR/cache/channels/`. Several workers can share a node. `scripts/run_channel_workers.sh` is the slurm script that starts one worker per GPU; workers claim rate limits as they go, so the slow ones near the phase transition are spread across workers.
2. `scripts/run_experiment.py --config <name>` runs the iterated learning simulation over the cached channels, writes the raw results to `data/raw/`, and post-processes them into `data/<name>-processed.csv`. `scripts/run_dirichlet_categorical.sh` is the slurm script for this step. It refuses to start if any channel the config needs is missing from the cache, since a cache miss would otherwise be recomputed inline on a single GPU. `scripts/run_model_variants.sh` runs the model variants reported in Appendix B.

The slurm scripts contain the account, partition, node, and environment paths we used, and will need editing for another cluster.

## Analysis and figures

Scripts in `scripts/analysis/` make the figures and compute the numbers reported in the paper. Most take command-line arguments via `tyro`; pass `--help` to see them.

- `make_plots.py` and `make_ssl_plots.py`: performance by generation and by rate for each model variant (Figures 3, 6, and B1 through B4), drawn from `data/*-processed.csv`.
- `visualize_one_matrix.py` (in `scripts/`) and `visualize_distortion_matrices.py`: channels for the small Beta-Bernoulli task (Figure 4).
- `make_channel_diagnostics_plot.py`: the channel diagnostics in Figure 5 (vocabulary size, expected confidence, and minimum confidence of channel outputs, by rate). Reading every cached channel takes a few minutes and only works on the machine holding the cache, so the script saves the table it plots from to `data/channel_diagnostics.csv`. Pass `--reuse-table` to redraw the figures from that table anywhere:

  ```bash
  python scripts/analysis/make_channel_diagnostics_plot.py --reuse-table
  ```

- `analyze_channel_marginals.py`: the outputs each channel sends most often, decoded into pseudocounts, saved to `data/channel_top_outputs.csv`. The specific channel outputs quoted in the paper come from this table.
- `analyze_channel_structure.py`: how often each belief survives transmission unchanged and which prototypes beliefs are rounded to, saved to `data/channel_structure.csv`.
- `verify_channel_convergence.py`: an independent check, in numpy, that every cached channel sits at the Blahut-Arimoto fixed point, saved to `data/channel_convergence_check.csv`.
- `make_beta_kl_plot.py` and `visualize_dirichlets.py`: illustrations of Beta and Dirichlet distributions.

The channels themselves are several hundred megabytes each and are not included in this repository; the tables above are what we computed from them.

### Worked example

`scripts/analysis/worked_example_densities.py` prints the density, and base-2 log density, that Bob's posterior assigns to the true probability at each of the three channel rates in the paper's worked example. With the paper's `p = 0.9`, the log densities are −0.04, 0.98, and 1.28.

`scripts/analysis/make_worked_example_plots.py` generates the components of Figure 2, the Beta-belief example with true probability `p = 0.9` and two observed heads:

```bash
python scripts/analysis/make_worked_example_plots.py
```

The script writes separate transparent SVG, PDF, and 300-dpi PNG files to
`figures/worked_example/` for assembly in Keynote. Belief and score canvases are 4 × 3 inches
(1200 × 900 pixels for PNG). SVG text is stored as paths to preserve its appearance
when imported. Belief panels show the filled curve and true-probability marker,
with minimal labels `x` and `p(x)`, endpoint numbers 0 and 1, and a light horizontal
baseline. They have no titles or vertical tick labels.
They share identical scales and margins; resize them equally to keep the density
comparisons meaningful. The separate score plot retains its labels.

| Filename stem | Belief or comparison |
| --- | --- |
| `alice_prior` | Beta(1, 1) |
| `alice_posterior` | Beta(2, 1), after heads |
| `bob_zero_prior` | Beta(2, 2), received belief |
| `bob_zero_posterior` | Beta(3, 2), after heads |
| `bob_intermediate_prior` | Beta(4, 2), received belief |
| `bob_intermediate_posterior` | Beta(5, 2), after heads |
| `bob_high_prior` | Beta(2, 1), received belief |
| `bob_high_posterior` | Beta(3, 1), after heads |
| `bob_scores` | Bob's posterior log₂ density at the true probability |
| `alice`, `bob` | Distinct stick figures, gesturing toward each other |
| `transmission_zero` | Strongly disrupted arrow with a break in the middle |
| `transmission_intermediate` | Moderately wavy arrow |
| `transmission_high` | Clean, straight arrow |
| `update_heads` | A coin marked H above an arrow, representing an observed-heads update |

The people and arrows are unlabeled ink sketches, exported in the same three transparent
formats. People use 2 × 3 inch canvases (600 × 900 pixels); arrows use 4 × 1.2 inch canvases
(1200 × 360 pixels). All arrows point right and have matching dimensions. Their waves
are deterministic artistic cues, not simulated signals or calibrated noise levels; the
zero-rate break indicates that no source-dependent information gets through.

Place `update_heads` between each prior and posterior, for Alice and for all three Bob
rows. H denotes the observed heads outcome: the update adds one to the Beta alpha
parameter. Its canvas is 2 × 1.5 inches (600 × 450 pixels), with the same transparent
SVG, PDF, and PNG exports as the other components.

Priors are green and posteriors are orange, using the existing Beta visualization's
palette. The dashed vertical line marks the true probability. The serif font is
Matplotlib's bundled DejaVu Serif. `beliefs.csv` records every panel's Beta parameters,
true probability, density, and base-2 log density. At `p = 0.9`, Bob's scores round to
−0.04, 0.98, and 1.28. These are log **densities**, not log probabilities.

The received beliefs are the specified illustrative outcomes; this script does not
compute an optimal channel, numerical channel rates, or the probability of receiving
each belief. The rate labels in the score panel are categories. Alice's and Bob's
posteriors are calculated by adding one heads observation to their respective priors.

Use `--output-dir figures/my_example` to change the destination (relative paths resolve
from the repository root), or `--true-p 0.7` to move the truth marker and recompute scores.
Changing the truth leaves the two observed heads and received beliefs fixed; the score
ordering need not remain the same. The script rejects nonfinite values and endpoints
because the score plot requires finite log densities. Rerunning replaces files with the
same names.

## Data

- `data/<config>-processed.csv`: processed simulation results for each configuration in `configs/`, with the expected score of learners in every generation for every rate limit.
- `data/channel_diagnostics.csv`, `data/channel_top_outputs.csv`, `data/channel_structure.csv`, and `data/channel_convergence_check.csv`: the tables written by the channel-analysis scripts above, computed from the cached channels used for the main results.

## Tests

The tests cover the information-theoretic utilities, the Blahut-Arimoto stopping rule, channel caching, the channel analysis code, and the figure scripts. Run them from the repository root:

```bash
MPLBACKEND=Agg python -m pytest tests -q
```

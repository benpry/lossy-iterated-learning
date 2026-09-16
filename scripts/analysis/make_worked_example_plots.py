"""Export separate Beta-belief panels for assembling the worked example in Keynote.

The received beliefs are the illustrative outcomes specified in the example. This
script does not solve a rate-distortion problem or estimate transmission probabilities.
Scores are base-2 log *densities*, not log probabilities of an exact continuous value.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import tyro
from pyprojroot import here
from scipy.stats import beta


@dataclass
class Args:
    output_dir: Path = here("figures/worked_example")
    """Destination for SVG, PDF, PNG, and the numerical CSV; relative paths use the repo root."""
    true_p: float = 0.9
    """True probability of heads; both observed outcomes remain heads."""


def belief_table(true_p: float) -> pd.DataFrame:
    """Record each belief, updating its alpha by one after an observed heads."""
    if not np.isfinite(true_p) or not 0 < true_p < 1:
        raise ValueError("true_p must be finite and strictly between 0 and 1")
    priors = [
        ("alice", "Alice", 1, 1),
        ("bob_zero", "Bob · zero rate", 2, 2),
        ("bob_intermediate", "Bob · intermediate rate", 4, 2),
        ("bob_high", "Bob · high rate", 2, 1),
    ]
    rows = []
    for component, label, a, b in priors:
        for stage, heads in [("prior", 0), ("posterior", 1)]:
            alpha = a + heads
            rows.append(
                {
                    "component": f"{component}_{stage}",
                    "label": label,
                    "stage": stage,
                    "alpha": alpha,
                    "beta": b,
                    "true_p": true_p,
                    "density_at_true_p": beta.pdf(true_p, alpha, b),
                    "log2_density_at_true_p": beta.logpdf(true_p, alpha, b) / np.log(2),
                }
            )
    return pd.DataFrame(rows)


def make_belief_plot(belief: pd.Series) -> plt.Figure:
    """Draw a belief with minimal axis labels on a shared scale for Keynote."""
    figure, ax = plt.subplots(figsize=(4, 3))
    # Fixed margins preserve identical canvas and axis dimensions across exports.
    figure.subplots_adjust(left=0.18, right=0.95, bottom=0.19, top=0.94)
    p = np.linspace(0, 1, 1001)
    density = beta.pdf(p, belief.alpha, belief.beta)
    color = "#66c2a5" if belief.stage == "prior" else "#fc8d62"
    ax.plot(p, density, color=color, linewidth=2.5)
    ax.fill_between(p, density, color=color, alpha=0.3)
    ax.axvline(belief.true_p, color="0.6", linestyle="--", linewidth=4.8)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 3.2),
        xticks=[0, 1],
        yticks=[],
    )
    ax.set_xticklabels(["0", "1"])
    ax.tick_params(axis="x", length=0, pad=6, labelsize=18)
    ax.set_xlabel("x", fontsize=21, fontstyle="italic")
    ax.xaxis.set_label_coords(0.5, -0.06)
    ax.set_ylabel("p(x)", fontsize=21, fontstyle="italic", rotation=90)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set(color="0.65", linewidth=0.8)
    return figure


def make_score_plot(beliefs: pd.DataFrame) -> plt.Figure:
    """Compare the three realized posteriors without implying numerical channel rates."""
    posteriors = beliefs[beliefs["component"].str.match("bob_.*_posterior")]
    scores = posteriors["log2_density_at_true_p"].to_numpy()
    figure, ax = plt.subplots(figsize=(4, 3))
    figure.subplots_adjust(left=0.2, right=0.95, bottom=0.22, top=0.78)
    bars = ax.bar(["Zero", "Intermediate", "High"], scores, color="#fc8d62", width=0.6)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.bar_label(
        bars, labels=[f"{score:.2f}".replace("-", "−") for score in scores], padding=5
    )
    # Include zero and leave space for labels, even when true_p changes the ranking.
    lower, upper = min(0, scores.min()), max(0, scores.max())
    margin = 0.2 * max(upper - lower, 1)
    ax.set(
        ylim=(lower - margin, upper + margin),
        xlabel="Channel rate",
        ylabel="Posterior log₂ density",
        title=f"Bob's score at true p = {beliefs['true_p'].iloc[0]:g}",
    )
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def make_person_plot(person: str) -> plt.Figure:
    """Draw a small ink figure; Alice gestures right and Bob gestures left."""
    if person not in ("alice", "bob"):
        raise ValueError(f"Unknown person: {person!r}; expected 'alice' or 'bob'")
    figure, ax = plt.subplots(figsize=(2, 3))
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    ax.set(xlim=(0, 1), ylim=(0, 1.5), aspect="equal")
    ax.set_axis_off()

    def stroke(x, y):
        # Mirror the pose so the two people can face one another in a left-to-right diagram.
        x = np.asarray(x)
        if person == "bob":
            x = 1 - x
        ax.plot(
            x,
            y,
            color="#343434",
            linewidth=2.4,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    angle = np.linspace(0, 2 * np.pi, 100)
    # A slight, deterministic irregularity gives the head a pen-drawn outline.
    radius = 0.15 * (1 + 0.025 * np.sin(3 * angle))
    stroke(0.48 + radius * np.cos(angle), 1.18 + radius * np.sin(angle))
    stroke([0.48, 0.47, 0.49], [1.03, 0.79, 0.53])  # Torso.
    stroke([0.47, 0.30, 0.25], [0.94, 0.77, 0.62])  # Resting arm.
    stroke([0.48, 0.67, 0.83], [0.94, 0.80, 0.92])  # Gesturing arm.
    stroke([0.49, 0.39, 0.30, 0.24], [0.53, 0.31, 0.13, 0.13])
    stroke([0.49, 0.57, 0.66, 0.73], [0.53, 0.32, 0.13, 0.12])
    if person == "alice":
        # A small bun distinguishes Alice even when the figures are scaled down.
        stroke(0.32 + 0.065 * np.cos(angle), 1.32 + 0.065 * np.sin(angle))
    else:
        stroke([0.36, 0.40, 0.44, 0.50, 0.57], [1.28, 1.36, 1.32, 1.37, 1.30])
    return figure


def make_transmission_plot(rate: str) -> plt.Figure:
    """Draw a conceptual noise cue, not a simulated signal or calibrated channel rate."""
    styles = {"zero": (0.28, 13), "intermediate": (0.12, 5), "high": (0.0, 0)}
    if rate not in styles:
        raise ValueError(
            f"Unknown channel rate: {rate!r}; expected one of {tuple(styles)}"
        )
    amplitude, cycles = styles[rate]
    figure, ax = plt.subplots(figsize=(4, 1.2))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    t = np.linspace(0, 1, 1001)
    x = 0.05 + 0.90 * t
    # Taper the disturbance to zero at both ends. Fixed waves make exports reproducible.
    disturbance = np.sin(2 * np.pi * cycles * t) + 0.3 * np.sin(2 * np.pi * 23 * t)
    y = 0.5 + amplitude * np.sin(np.pi * t) ** 2 * disturbance
    if rate == "zero":
        # A broken shaft suggests that no source-dependent information gets through.
        y[(t > 0.46) & (t < 0.54)] = np.nan
    ax.plot(x, y, color="#343434", linewidth=3.3, solid_capstyle="round")
    ax.plot(
        [0.86, 0.95, 0.86],
        [0.65, 0.5, 0.35],
        color="#343434",
        linewidth=3.3,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    return figure


def make_update_plot() -> plt.Figure:
    """Draw a heads coin above the arrow taking a prior to its updated posterior."""
    figure, ax = plt.subplots(figsize=(2, 1.5))
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    ax.set(xlim=(0, 2), ylim=(0, 1.5), aspect="equal")
    ax.set_axis_off()
    ink = "#343434"
    center = (1, 1)
    ax.add_patch(
        Circle(center, 0.32, facecolor="#f5e6bf", edgecolor=ink, linewidth=2.2)
    )
    ax.add_patch(Circle(center, 0.26, fill=False, edgecolor=ink, linewidth=0.8))
    ax.text(*center, "H", ha="center", va="center", fontsize=22, color=ink)
    ax.plot(
        [0.18, 1.82], [0.32, 0.32], color=ink, linewidth=2.2, solid_capstyle="round"
    )
    ax.plot(
        [1.60, 1.82, 1.60],
        [0.47, 0.32, 0.17],
        color=ink,
        linewidth=2.2,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    return figure


def save_component(figure: plt.Figure, output_dir: Path, name: str) -> None:
    """Export a fixed-size transparent canvas, closing it even if an export fails."""
    try:
        for extension in ("svg", "pdf", "png"):
            figure.savefig(
                output_dir / f"{name}.{extension}", transparent=True, dpi=300
            )
    finally:
        plt.close(figure)


def main(args: Args) -> None:
    beliefs = belief_table(args.true_p)
    output_dir = here(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Local style settings avoid changing other plots when this module is imported.
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    ):
        for _, belief in beliefs.iterrows():
            save_component(make_belief_plot(belief), output_dir, belief.component)
        save_component(make_score_plot(beliefs), output_dir, "bob_scores")
        save_component(make_update_plot(), output_dir, "update_heads")
        for person in ("alice", "bob"):
            save_component(make_person_plot(person), output_dir, person)
        for rate in ("zero", "intermediate", "high"):
            save_component(
                make_transmission_plot(rate), output_dir, f"transmission_{rate}"
            )
    beliefs.to_csv(output_dir / "beliefs.csv", index=False)
    print(
        f"Saved 15 components as SVG, PDF, and PNG, plus beliefs.csv, to {output_dir}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

"""
This file fills the channel cache, one worker per GPU.

Computing a channel is by far the most expensive part of a sweep, every beta is independent of every
other, and the results go to a cache on local disk. So several workers on the same node can divide
the betas between them and the sweep afterwards becomes a series of cache hits.

Workers claim betas as they go rather than taking a fixed share, because the betas near the phase
transition cost thousands of times more than the ones away from it.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import tyro
from box import Box
from pyprojroot import here

from src.channel_analysis import SOURCE_DENSITIES, ChannelSpec, channel_cache_path
from src.experiment import Experiment, get_distortion_matrix


@dataclass
class Args:
    # name of a config file in configs/
    config: str
    # which worker this is, and how many there are in total
    worker_index: int = 0
    n_workers: int = 1
    cache_dir: Path = Path(os.environ.get("SCR_ROOT_DIR", "/scr/benpry")) / "cache" / "channels"
    # refuse to run on the CPU, where these channels take days rather than hours
    require_gpu: bool = True


def claim(cache_path: Path):
    """
    Take exclusive responsibility for computing one channel.

    Returns the path of the claim file, or None if another worker already holds it. Creating the file
    exclusively is atomic, so two workers on the same node can never pick the same beta.
    """
    claim_path = cache_path.with_suffix(".claim")
    try:
        os.close(os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return None

    return claim_path


def main(args: Args):
    if args.require_gpu and jax.default_backend() not in ("gpu", "cuda"):
        raise RuntimeError(
            f"jax sees no GPU, only {jax.devices()}. These channels take days on the CPU; pass "
            f"--no-require-gpu to do it anyway."
        )

    cfg = Box.from_yaml(filename=here(f"configs/{args.config}.yaml"))
    betas = np.logspace(-4, cfg.max_beta, num=cfg.n_betas, base=2.0)
    os.makedirs(args.cache_dir, exist_ok=True)

    print(
        f"worker {args.worker_index}/{args.n_workers} on {jax.devices()} "
        f"for {args.config}: {cfg.n_betas} betas, source={cfg.source_distribution}",
        flush=True,
    )

    distortion_matrix = get_distortion_matrix(
        cfg.max_val, cfg.dimension, cfg.distortion_metric, cfg.n_observations
    )
    experiment = Experiment(
        true_probs=cfg.true_probs,
        distortion_metric=cfg.distortion_metric,
        distortion_matrix=distortion_matrix,
        n_generations=cfg.n_generations,
        n_observations=cfg.n_observations,
        max_param_val=cfg.max_val,
        convergence_tolerance=cfg.convergence_tolerance,
        max_channel_iters=cfg.max_channel_iters,
        channel_cache_dir=args.cache_dir,
    )

    computed, failed = [], []
    for beta in betas:
        spec = ChannelSpec(
            dimension=cfg.dimension,
            max_val=cfg.max_val,
            beta=float(beta),
            source_distribution=cfg.source_distribution,
            distortion_metric=cfg.distortion_metric,
        )
        cache_path = channel_cache_path(spec, args.cache_dir)
        if cache_path.exists():
            continue

        claim_path = claim(cache_path)
        if claim_path is None:
            continue

        try:
            print(f"worker {args.worker_index}: computing beta={beta:.6f}", flush=True)
            _, _, rate, distortion, iters, remaining = experiment.compute_channel(
                beta=beta,
                source_distribution_fn=SOURCE_DENSITIES[cfg.source_distribution],
                source_distribution_str=cfg.source_distribution,
            )
            print(
                f"worker {args.worker_index}: beta={beta:.6f} done in {int(iters):,} iterations, "
                f"rate={float(rate):.5f} distortion={float(distortion):.5f} "
                f"remaining={float(remaining):.3g}",
                flush=True,
            )
            computed.append(float(beta))
        except Exception as error:
            # keep the worker alive so one stubborn beta does not idle a GPU, but say so loudly and
            # leave a non-zero exit code behind
            print(f"worker {args.worker_index}: beta={beta:.6f} FAILED: {error}", flush=True)
            failed.append(float(beta))
        finally:
            # release the claim either way, so a later run can retry
            claim_path.unlink(missing_ok=True)

    print(
        f"worker {args.worker_index}: computed {len(computed)} channels, {len(failed)} failed",
        flush=True,
    )
    if failed:
        raise SystemExit(f"worker {args.worker_index} failed on betas {failed}")


if __name__ == "__main__":
    main(tyro.cli(Args))

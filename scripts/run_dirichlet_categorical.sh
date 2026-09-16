#!/bin/zsh
#SBATCH --job-name=dirichlet_categorical
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
# The channels this reads are 592 MB each and live in /scr, which is a local disk on each node.
# scripts/run_channel_workers.sh writes them on cocoflops-hgx-1, so the sweep has to run there too.
# This matters more than it looks: cocoflops1 has its own /scr holding an older, partial set of
# channels, and a stale channel is not a loud failure. It records how far it was from converging,
# the sweep accepts it on that record, and the records written before the stopping rule measured
# decay across a window are the ones that cannot be trusted.
#SBATCH --nodelist=cocoflops-hgx-1
# absolute, so the logs land in the same place no matter which directory sbatch is run from. The
# config is part of the job rather than the filename, so the job id keeps concurrent sweeps apart.
#SBATCH --output=/sailhome/benpry/lossy-iterated-learning/scripts/slurm-output/experiment-%j.out
#SBATCH --error=/sailhome/benpry/lossy-iterated-learning/scripts/slurm-output/experiment-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Load any necessary modules
source ~/.zshrc

# Change to the working directory
cd ~/lossy-iterated-learning

# activate the uv environment
source /scr/benpry/uv/rd-culture/bin/activate

# the repo is not pip installed in every environment, and running a script from scripts/ puts that
# directory on sys.path rather than the repo root
export PYTHONPATH=$PWD

CONFIG=${1:-dirichlet_categorical}
echo "host: $(hostname), config: $CONFIG"

# Refuse to start if jax cannot see the GPU. Blahut-Arimoto now runs to convergence rather than for
# a fixed number of iterations, so falling back to the CPU means days of compute instead of hours.
python -c "
import sys
import jax
if jax.default_backend() not in ('gpu', 'cuda'):
    sys.exit(f'jax sees no GPU, only {jax.devices()}. Refusing to run this on the CPU.')
print(f'running on {jax.devices()}')
" || exit 1

# Stop now if any channel this config needs is absent from the cache, rather than after the sweep
# has quietly started recomputing them one at a time on a single GPU. That is what it would do: a
# missing channel is a cache miss, and a cache miss is computed inline. The betas near the phase
# transition take four million iterations each, so a sweep that starts with an empty cache runs for
# days. Only existence is checked here -- reading the convergence each channel recorded would mean
# unpickling eighteen gigabytes before the job starts.
python -c "
import os
import socket
import sys
from pathlib import Path

import numpy as np
from box import Box
from pyprojroot import here
from src.channel_analysis import ChannelSpec, channel_cache_path

cfg = Box.from_yaml(filename=here('configs/$CONFIG.yaml'))
betas = np.logspace(-4, cfg.max_beta, num=cfg.n_betas, base=2.0)

# the directory Experiment falls back to when run_experiment.py does not name one
cache_dir = Path(os.environ['SCR_ROOT_DIR']) / 'cache' / 'channels'
missing = [
    float(beta)
    for beta in betas
    if not channel_cache_path(
        ChannelSpec(
            dimension=cfg.dimension,
            max_val=cfg.max_val,
            beta=float(beta),
            source_distribution=cfg.source_distribution,
            distortion_metric=cfg.distortion_metric,
        ),
        cache_dir,
    ).exists()
]
if missing:
    sys.exit(
        f'{len(missing)} of {cfg.n_betas} channels for $CONFIG are not in {cache_dir} on '
        f'{socket.gethostname()}: betas {[round(b, 3) for b in missing]}. '
        f'Run scripts/run_channel_workers.sh $CONFIG first.'
    )
print(f'all {cfg.n_betas} channels for $CONFIG are cached in {cache_dir}')
" || exit 1

python scripts/run_experiment.py --config "$CONFIG"

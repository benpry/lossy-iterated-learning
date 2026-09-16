#!/bin/zsh
#SBATCH --job-name=channel_workers
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
# hgx-1 has eight A100s. The Blahut-Arimoto iteration is memory bandwidth bound, so the A100s are
# worth roughly three times an A40 each, and /scr is local to the node so all eight workers share
# one channel cache.
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --gres=gpu:a100:8
#SBATCH --output=/sailhome/benpry/lossy-iterated-learning/scripts/slurm-output/channel-workers-%j.out
#SBATCH --error=/sailhome/benpry/lossy-iterated-learning/scripts/slurm-output/channel-workers-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00

source ~/.zshrc
cd ~/lossy-iterated-learning

source /scr/benpry/uv/rd-culture/bin/activate

# the repo is not pip installed in every environment, and running a script from scripts/ puts that
# directory on sys.path rather than the repo root
export PYTHONPATH=$PWD

# fail immediately, with a clear message, rather than after the queue wait
python -c "import src.channel_analysis, jax, tyro, box" || {
  echo "the environment cannot import what the workers need"; exit 1
}

CONFIG=${1:-dirichlet_categorical}

# One worker per GPU we were actually given, rather than per GPU we asked for. The eight in the
# #SBATCH line above are what this wants; passing --gres=gpu:a100:4 to sbatch overrides it and takes
# half the node when someone else holds the other half, which beats waiting days for all eight.
# Slurm points CUDA_VISIBLE_DEVICES at the allocation, so nvidia-smi lists exactly our share.
N_WORKERS=$(nvidia-smi --list-gpus | wc -l)
if [ "$N_WORKERS" -lt 1 ]; then
  echo "no GPUs in this allocation"; exit 1
fi

echo "host: $(hostname), config: $CONFIG, workers: $N_WORKERS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Channels cached before the stopping rule measured decay across a window were stopped by a ratio of
# two consecutive steps, which is not measurable, so several of them settled short of the tolerance
# they claim. They cannot be told apart from good ones by anything in the file, so retire the lot
# rather than let a worker skip a beta because a stale file happens to sit at its path.
CACHE=$SCR_ROOT_DIR/cache/channels
RETIRED=$SCR_ROOT_DIR/cache/channels_single_step_rule
if [ ! -f "$CACHE/.window_stopping_rule" ]; then
  if [ -d "$RETIRED" ]; then
    echo "both $CACHE and $RETIRED exist: move or remove one by hand before rerunning"
    exit 1
  fi
  [ -d "$CACHE" ] && mv "$CACHE" "$RETIRED" && echo "retired old channels to $RETIRED"
  mkdir -p "$CACHE"
  touch "$CACHE/.window_stopping_rule"
fi

# Build the distortion matrix once up front. It is cached on local disk, so otherwise all eight
# workers would build the same one at the same time on startup.
python -c "
from box import Box
from pyprojroot import here
from src.experiment import get_distortion_matrix
cfg = Box.from_yaml(filename=here('configs/$CONFIG.yaml'))
get_distortion_matrix(cfg.max_val, cfg.dimension, cfg.distortion_metric, cfg.n_observations)
print('distortion matrix ready for $CONFIG')
" || exit 1

# Time the inner loop before committing the node to hours of it. The stopping rule keeps a copy of
# the channel from the start of the current window, and if that copy were written on every iteration
# instead of once per window it would roughly double the memory traffic of a step. A step of this
# size takes about 3 ms on one of these A100s; well above that means stop and look.
python -c "
import time
import jax.numpy as jnp
from box import Box
from pyprojroot import here
from src.experiment import get_distortion_matrix
from src.info_theory import blahut_arimoto

cfg = Box.from_yaml(filename=here('configs/$CONFIG.yaml'))
distortion = jnp.asarray(
    get_distortion_matrix(cfg.max_val, cfg.dimension, cfg.distortion_metric, cfg.n_observations)
)
source_p = jnp.ones(distortion.shape[0]) / distortion.shape[0]

# a tolerance of zero means it runs the whole budget rather than stopping partway through the timing
def run(n_iters):
    channel, *_ = blahut_arimoto(
        source_p, distortion, 0.5, distortion.shape[1], max_iters=n_iters, tolerance=0.0
    )
    return channel.block_until_ready()

run(10)
n_timed = 2000
start = time.time()
run(n_timed)
print(f'inner loop: {(time.time() - start) * 1000 / n_timed:.2f} ms/iteration', flush=True)
" || exit 1

# one worker per GPU, each seeing only its own device so jax does not try to share
pids=()
for worker in $(seq 0 $((N_WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES=$worker python scripts/compute_channels.py \
    --config "$CONFIG" --worker-index "$worker" --n-workers "$N_WORKERS" &
  pids+=($!)
done

# wait on each worker individually, so a job whose workers all died does not report success
failures=0
for pid in $pids; do
  wait $pid || failures=$((failures + 1))
done

if [ $failures -gt 0 ]; then
  echo "$failures of $N_WORKERS workers failed"
  exit 1
fi
echo "all $N_WORKERS workers finished cleanly"

#!/bin/zsh
#SBATCH --job-name=model_variants
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1,cocoflops2
#SBATCH --output=slurm-output/model-variants.out
#SBATCH --error=slurm-output/model-variants.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Load any necessary modules
source ~/.zshrc

cd ~/rate-distortion-culture

conda activate rd-culture
# python scripts/run_experiment.py --config dirichlet_categorical_longlife
# python scripts/run_experiment.py --config dirichlet_categorical_decreasing
# python scripts/run_experiment.py --config dirichlet_categorical_closertrueprobs
# python scripts/run_experiment.py --config dirichlet_categorical_farthertrueprobs
python scripts/run_experiment.py --config dirichlet_categorical_ssl

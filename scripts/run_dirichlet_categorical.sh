#!/bin/zsh
#SBATCH --job-name=dirichlet_categorical
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1,cocoflops2
#SBATCH --output=slurm-output/dirichlet-categorical.out
#SBATCH --error=slurm-output/dirichlet-categorical.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Load any necessary modules
source ~/.zshrc

# Change to the working directory
cd ~/rate-distortion-culture

conda activate rd-culture
python scripts/run_experiment.py --config dirichlet_categorical

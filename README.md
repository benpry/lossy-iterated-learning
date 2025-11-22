# lossy-iterated-learning
Code and data for the paper "Lossy communication constraints iterated learning" by Prystawski, Arumugam, and Goodman.

To reproduce our results, you should first create a conda environment and install the dependencies in `requirements.txt`. Next, you should install the source code for this package.

```bash
conda create -n rd-culture python=3.12 pip
conda activate rd-culture
pip install -r requirements.txt
pip install -e .
```

The `src/` directory contains the core code and utilities for running simulations. Most of this is in `experiment.py`. `info_theory.py` contains information theoretic utilities, including our implementation of the Blahut-Arimoto algorithm. `utils.py` contains most utilities, and `process_results.py` contains the code for computing metrics like expected scores from raw vectors of proportions.

You can actually run the experiments using `run_experiment.py`, which you can configure by passing the name of one of the configuration files in the `configs/` directory. There are two slurm scripts that run different configurations in this directory too.
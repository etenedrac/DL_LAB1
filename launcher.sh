#!/bin/bash

#SBATCH --job-name="test_mnist"

#SBATCH --workdir=.

#SBATCH --output=mnist_%j.out

#SBATCH --error=mnist_%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=00:02:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python bsc.py

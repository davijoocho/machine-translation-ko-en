#!/bin/bash

#SBATCH --output="train.out"
#SBATCH --error="train.out"
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=3
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodelist=c[17-20]

module purge
module load anaconda3
source activate transformer

export CUDA_HOME=$CONDA_PREFIX/lib:$CUDA_HOME
export PATH=$CUDA_HOME:$CONDA_PREFIX/lib:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun python3 code/train.py


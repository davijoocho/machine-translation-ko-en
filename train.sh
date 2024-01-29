#!/bin/bash

#SBATCH --output="train.out"
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load miniconda
conda activate transformer

python3 code/en_ko_translation.py


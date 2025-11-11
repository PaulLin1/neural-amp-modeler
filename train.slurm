#!/bin/bash
#SBATCH --job-name=gpcv
#SBATCH --output=it%j.out
#SBATCH --error=it%j.err
#SBATCH --partition=h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

source venv/bin/activate

python train.py
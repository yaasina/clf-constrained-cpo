#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1

export WANDB_MODE=offline
export WANDB_API_KEY=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

cd ~/projects/def-danielac/sarvan13/clf-constrained-cpo/src/
module purge
module load python/3.10.13
module load mujoco
module load StdEnv/2023
module load openblas
source ~/MujocoENV/bin/activate

python -u main.py

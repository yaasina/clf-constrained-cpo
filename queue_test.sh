#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac

cd ~/projects/def-danielac/sarvan13/clf-constrained-cpo/
module purge
module load python/3.10.13
module load mujoco
module load StdEnv/2023
module load openblas
source ~/MujocoENV/bin/activate

python -u plot_pend_rew.py

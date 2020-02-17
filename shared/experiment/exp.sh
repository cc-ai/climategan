#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --gres=:1
#SBATCH --mem=32
#SBATCH -o /network/tmp1/victor/omnigan/experiments/run/slurm-%j.out
#SBATCH -p unkillable



cd /network/home/$USER/omnigan

module load anaconda/3 >/dev/null 2>&1
. "$CONDA_ACTIVATE"
conda deactivate
conda activate omnigan

echo "Starting job"

python train.py --config=/network/tmp1/victor/omnigan/experiments/run/config.yaml  --exp_desc="Training d only - 1" 

echo 'done'
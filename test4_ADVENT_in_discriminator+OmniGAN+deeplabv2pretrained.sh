#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=8                     # Ask for 2 CPUs
#SBATCH --gres=gpu:titanrtx:1                          # Ask for 1 GPU
#SBATCH --mem=48G                             # Ask for 10 GB of RAM
#SBATCH -o /network/tmp1/tianyu.zhang/slurm-%j.out  # Write the log on tmp1
​
module load anaconda/3
source $CONDA_ACTIVATE
conda activate myenv

​
# 1. Load your environment
# conda init bash
​
# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
# python train_CCAI.py --cfg ./configs/advent.yml
curl https://notify.run/JqEa7hwTF0Ki8dl1 -d "message goes here"
python train.py --config ./config/trainer/maskgen_v0.yaml

# 4. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/output /network/tmp1/tianyu.zhang/


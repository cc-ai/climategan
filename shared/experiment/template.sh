#!/bin/bash
#SBATCH --cpus-per-task={{cpu}}
#SBATCH --gres={{gpu}}
#SBATCH --mem={{mem}}
#SBATCH --time={{time}}
#SBATCH -o {{output_path}}/slurm-%j.out
{{main_partition}}

{{zip_command}}

{{cp_unzip_command}}

cd /network/home/$USER/omnigan

module load anaconda/3 >/dev/null 2>&1
. "$CONDA_ACTIVATE"
conda deactivate
conda activate omnigan

echo "Starting job"

python train.py {{config}} {{no_comet}} {{exp_desc}} {{dev_mode}}  {{tags}}

echo 'done'

#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --output={output}

module purge

{modules}

{conda}

export PYTHONUNBUFFERED=1

cd {codeloc}

echo "Currently using:"
echo $(which python)
echo "in:"
echo $(pwd)
echo "sbatch file name: $0"

python train.py {train_args}
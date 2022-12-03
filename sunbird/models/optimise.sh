#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi_g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH -q regular
#SBATCH -t 07:00:00
#SBATCH --array=3-4

set -e
DS=$((SLURM_ARRAY_TASK_ID))

python optimise.py --model_dir /global/homes/e/epaillas/code/sunbird/trained_models/optimise_s2/ds"$DS" --statistic ds"$DS" --accelerator gpu --apply_s2 True

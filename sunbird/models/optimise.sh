#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH --array=4-4

set -e
export SLURM_CPU_BIND="cores"
# DS=$((SLURM_ARRAY_TASK_ID))
DS=4

CORR=cross
python optimise.py --abacus_dataset wideprior_AB --model_dir /global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/wideprior_AB/log_ds"$DS"_"$CORR" --statistic density_split_"$CORR" --normalize_outputs true --normalize_inputs true --s2_outputs False --select_quintiles "$DS" --select_multipoles 0 1 --slice_s 0.7 151 --accelerator gpu
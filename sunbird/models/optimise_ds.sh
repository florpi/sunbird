#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH --array=3-4

set -e
export SLURM_CPU_BIND="cores"
DS=$((SLURM_ARRAY_TASK_ID))
ROOT_DIR=/global/homes/e/epaillas/pscratch/sunbird
CORR=cross
ACT_FN=SiLU

python optimise.py \
--abacus_dataset wideprior_AB \
--model_dir "$ROOT_DIR"/trained_models/enrique/wideprior_AB/"$ACT_FN"/ds"$DS"_"$CORR" \
--statistic density_split_"$CORR" \
--select_quintiles "$DS" \
--select_multipoles 0 1 \
--slice_s 0.7 151 \
--accelerator gpu \
--act_fn "$ACT_FN"
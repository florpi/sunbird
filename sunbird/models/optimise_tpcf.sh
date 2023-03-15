#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH -q regular
#SBATCH -t 04:00:00

set -e
export SLURM_CPU_BIND="cores"
ACT_FN=SiLU

python optimise.py \
--abacus_dataset wideprior_AB \
--model_dir /global/homes/e/epaillas/pscratch/sunbird/trained_models/enrique/wideprior_AB/"$ACT_FN"/s2/tpcf \
--statistic tpcf \
--select_multipoles 0 1 \
--slice_s 0.7 151 \
--accelerator gpu \
--act_fn "$ACT_FN" \
--s2_outputs true \
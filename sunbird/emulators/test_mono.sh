#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --output=/global/homes/t/tsfraser//%x-%j.out
#SBATCH --error=/global/homes/t/tsfraser/%x-%j.error
#SBATCH --mail-user=tsfraser@uwaterloo.ca
#SBATCH -q regular
#SBATCH -t 04:00:00

abacus_dataset=voidprior
loss=mae
model_dir=/global/homes/t/tsfraser/voids_sunbird/sunbird/sunbird/trained_models/voids/
run_name=optimal_settings
statistic=voids
train_test_split=/global/homes/t/tsfraser/voids_sunbird/sunbird/data/train_test_split.json


python /global/homes/t/tsfraser/voids_sunbird/sunbird/sunbird/emulators/train.py \
    --model_dir "$model_dir" \
    --run_name "$run_name" \
    --abacus_dataset "$abacus_dataset" \
    --statistic "$statistic" \
    --select_multipoles 0 \
    --slice_s 0 50 \
    --train_test_split "$train_test_split" \
    --loss "$loss" \
    --accelerator gpu \
    # --output_transforms Normalize \
    # --independent_avg_scale True \
    # --n_hidden 639 477 444 \
    # --weight_decay 0.007824703383429452 \
    # --learning_rate 0.004227486931154342 \
    # --dropout_rate 0.017416363761991755 \

#--normalize_outputs true \
#--normalize_inputs true \

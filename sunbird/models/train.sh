#!/bin/bash

set -e

DS=0
python train.py --model_dir /global/homes/e/epaillas/code/sunbird/trained_models/enrique/ds"$DS" --statistic ds"$DS" --batch_size 81 --learning_rate 0.06976434750083169 --n_hidden 84 --n_layers 4 --normalize true --normalize_inputs true --weight_decay 0.007973020871870054 --apply_s2 False

DS=1
python train.py --model_dir /global/homes/e/epaillas/code/sunbird/trained_models/enrique/ds"$DS" --statistic ds"$DS" --batch_size 167 --learning_rate 0.028587789506922916 --n_hidden 52 --n_layers 5 --normalize true --normalize_inputs true --weight_decay 0.00827226050163605 --apply_s2 False

DS=3
python train.py --model_dir /global/homes/e/epaillas/code/sunbird/trained_models/enrique/ds"$DS" --statistic ds"$DS" --batch_size 150 --learning_rate 0.02440618506088254 --n_hidden 34 --n_layers 5 --normalize true --normalize_inputs true --weight_decay 0.0013043993777777981 --apply_s2 False

DS=4
python train.py --model_dir /global/homes/e/epaillas/code/sunbird/trained_models/enrique/ds"$DS" --statistic ds"$DS" --batch_size 70 --learning_rate 0.013094307070946717 --n_hidden 32 --n_layers 5 --normalize true --normalize_inputs true --weight_decay 0.00746633847720323 --apply_s2 False

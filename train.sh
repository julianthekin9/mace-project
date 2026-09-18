#!/usr/bin/env bash
set -e

# 1) окружение (один раз)
pip install mace-torch

# 2) деление dataset.extxyz на train/valid/test
python prepare_data.py --input dataset.extxyz --outdir data \
    --valid-frac 0.1 --test-frac 0.1 --seed 42

# 3) обучение
mace_run_train --config config.yaml

# 4) оценка на test.xyz
python predict.py --model models/mace_energy_model.model --input data/test.xyz

#!/bin/sh

model_save_dir=/scratch/maths/dual/mt6170499/cls/model
preds_save_dir=/scratch/maths/dual/mt6170499/cls/preds
cache_dir=/scratch/maths/dual/mt6170499/.cache

python -u cls.py --model_name matscibert --lm_lrs 2e-5 --seeds 1 --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir

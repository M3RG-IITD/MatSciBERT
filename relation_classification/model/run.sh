#!/bin/sh

model_save_dir=model/
preds_save_dir=preds/
cache_dir=../.cache/

python -u relation_classification.py --model_name matscibert --lm_lrs 2e-5 --seeds 2 --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir

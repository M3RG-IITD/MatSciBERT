#!/bin/sh

model_save_dir=models/matscholar/
preds_save_dir=preds/
cache_dir=../.cache/

python -u ner.py --model_name matscibert --non_lm_lr 5e-4 --lm_lrs 5e-5 --seeds 1 --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture bert-crf --dataset_name matscholar 

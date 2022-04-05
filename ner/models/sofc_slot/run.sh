#!/bin/sh

model_save_dir=models/sofc_slot/
preds_save_dir=preds/
cache_dir=../.cache/

python -u ner.py --model_name matscibert --lm_lrs 2e-5 --seeds 2 --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture bert-crf --dataset_name sofc_slot --fold_num 1

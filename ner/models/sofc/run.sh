#!/bin/sh

model_save_dir=models/sofc/
preds_save_dir=preds/
cache_dir=../.cache/

python -u ner.py --model_name matscibert --lm_lrs 5e-5 --seeds 0 --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture bert-crf --dataset_name sofc --fold_num 1

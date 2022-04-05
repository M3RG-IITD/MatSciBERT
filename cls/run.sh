#!/bin/sh

model_save_dir=model/
preds_save_dir=preds/
cache_dir=../.cache/

for model_name in matscibert scibert bert; do
    echo $model_name
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir
done

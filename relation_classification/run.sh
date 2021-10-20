#!/bin/sh

model_save_dir=/scratch/maths/dual/mt6170499/relation_classification/model
preds_save_dir=/scratch/maths/dual/mt6170499/relation_classification/preds
cache_dir=/scratch/maths/dual/mt6170499/.cache

for model_name in matscibert scibert; do
    echo $model_name
    python -u relation_classification.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir
done

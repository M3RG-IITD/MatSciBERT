#!/bin/sh

root_dir=/home/maths/dual/mt6170499/scratch
out_dir=/home/maths/dual/mt6170499/scratch/cls/output

for model_name in scibert matscibert; do
    echo $model_name
    python -u cls.py --root_dir $root_dir --model_name $model_name --output_dir $out_dir
done

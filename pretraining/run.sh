#!/bin/sh

cache_dir=/scratch/maths/dual/mt6170499/.cache
train_file=/scratch/maths/dual/mt6170499/matscibert/data/train_corpus.txt
val_file=/scratch/maths/dual/mt6170499/matscibert/data/val_corpus.txt
train_norm_file=/scratch/maths/dual/mt6170499/matscibert/data/train_corpus_norm.txt
val_norm_file=/scratch/maths/dual/mt6170499/matscibert/data/val_corpus_norm.txt
model_save_dir=/scratch/maths/dual/mt6170499/matscibert/model

python -u normalize_corpus.py --train_file $train_file --val_file $val_file --output_train_norm_file $train_norm_file --output_val_norm_file $val_norm_file
python -u matscibert_pre_train.py --train_file $train_norm_file --val_file $val_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir

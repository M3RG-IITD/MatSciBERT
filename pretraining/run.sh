#!/bin/sh

root_dir=/home/maths/dual/mt6170499/scratch
train_file=/home/maths/dual/mt6170499/scratch/wwm/corpus_final/train_150000_corpus.txt
val_file=/home/maths/dual/mt6170499/scratch/wwm/corpus_final/val_150000_corpus.txt
train_norm_file=/home/maths/dual/mt6170499/scratch/wwm/corpus_final/train_150000_corpus_norm.txt
val_norm_file=/home/maths/dual/mt6170499/scratch/wwm/corpus_final/val_150000_corpus_norm.txt
model_save_dir=/home/maths/dual/mt6170499/scratch/wwm/output_dataset_final/scibert_uncased

python -u normalize_corpus.py --train_file $train_file --val_file $val_file --output_train_norm_file $train_norm_file --output_val_norm_file $val_norm_file
python -u matscibert_pre_train.py --root_dir $root_dir --train_file $train_norm_file --val_file $val_norm_file --output_dir $model_save_dir

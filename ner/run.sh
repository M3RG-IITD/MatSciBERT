#!/bin/sh

out_dir=/home/maths/dual/mt6170499/scratch/ner/output

for model_name in scibert matscibert; do
    for arch in bert bert-crf bert-bilstm-crf; do
        for dataset in sofc sofc_slot; do
            for fold in {1..5}; do
                echo $model_name $arch $dataset $fold
                python -u ner.py --model_name $model_name --architecture $arch --dataset_name $dataset --fold_num $fold --output_dir $out_dir
            done
        done

        echo $model_name $arch matscholar
        python -u ner.py --model_name $model_name --architecture $arch --dataset_name matscholar --output_dir $out_dir
        
    done
done

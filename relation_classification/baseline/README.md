# Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification

## Dataloading
```
python data/reader_create_data.py
python data/reader.py
python preprocess.py -lower
```

## Train
```bash
python -W ignore train.py -seed 0 -save_dir out_0/ -save_dir_cp out_cp_0/
```

## Reference
* [**Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification**](http://www.aclweb.org/anthology/P16-2034) (ACL 2016), P Zhou et al.
* zhijing-jin's [PyTorch implementation](https://github.com/zhijing-jin/pytorch_RelationExtraction_AttentionBiLSTM)

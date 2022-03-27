# Abstract Classification Baseline

## Dependencies
The code requires the following dependencies to run can be installed using the `conda` environment file provided:
```bash
conda env create --file environment.yaml
conda activate torch1.4
python -m spacy download en_core_web_sm
```

## Running code
```bash
python train.py --pool max --mode train --batch_size 16 --task glass_non_glass --epochs 15 --log 1 --customlstm 0 --seed 0
```
```bash
python train.py --pool att_max --mode train --batch_size 16 --task glass_non_glass --epochs 15 --log 1 --customlstm 0 --seed 0
```

## References
* [**Why and when should you pool? Analyzing Pooling in Recurrent Architectures**](https://www.aclweb.org/anthology/2020.findings-emnlp.410) (ACL 2020), Maini et al.

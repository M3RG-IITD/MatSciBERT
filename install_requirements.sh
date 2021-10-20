#!/bin/sh

conda create -n matscibert python=3.7.9
conda activate matscibert
conda install -y numpy==1.20.3 pandas==1.2.4 scikit-learn=0.23.2
conda install -y pytorch==1.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt

#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=nvcr-torch-1712

~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/comp_architectures/diet_baseline.json -s
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/comp_architectures/diet_fs_0.0001.json -s

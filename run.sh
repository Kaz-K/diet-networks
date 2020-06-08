#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=nvcr-torch-1712

~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures2/diet_fs_0.001_recon_0.001.json
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures2/diet_fs_0.001_recon_0.01.json
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures2/diet_fs_0.001_recon_0.1.json 

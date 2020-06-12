#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=nvcr-torch-1712

~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures/mlp_baseline.json
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures/mlp_fs.json
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures/diet_recon_0.1.json
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python visualize_emb.py -c config/comp_architectures/diet_fs_recon_0.1.json

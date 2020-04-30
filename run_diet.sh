#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=nvcr-torch-1712

~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/DietNetworks/diet_recon_0.1.json -s
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/DietNetworks/diet_recon_0.05.json -s
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/DietNetworks/diet_recon_0.005.json -s
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/DietNetworks/diet_recon_0.5.json -s
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/DietNetworks/diet_recon_1.0.json -s
~/.pyenv/versions/anaconda3-5.2.0/envs/pytorch/bin/python train_k_holds.py -c config/DietNetworks/diet_recon_5.0.json -s

import os
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
from functools import partial
import scipy.stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataio import get_k_hold_data_loader
from models import get_model
from utils import load_json
from utils import check_manual_seed
from utils import save_config
from utils import save_logs
from utils import save_models
from utils import load_model


def get_saved_model_path(config, study_name, checkpoint_epoch, i, n_splits,
                         sep='Jun'):
    saved_root_dir = config.save.output_root_dir
    saved_dir_path = None
    for dir_name in os.listdir(saved_root_dir):
        sep_index = re.search(sep, dir_name).start()
        saved_study_name = dir_name[:sep_index-1]
        if study_name == saved_study_name:
            saved_dir_path = os.path.join(saved_root_dir, dir_name)
            break

    assert saved_dir_path is not None

    pattern1 = 'epoch_' + str(checkpoint_epoch)
    pattern2 = '_' + str(i) + '_' + str(n_splits) + '.pth'

    target_model_name = None
    for model_name in os.listdir(saved_dir_path):
        if model_name.startswith(pattern1):
            if model_name.endswith(pattern2):
                target_model_name = model_name
                break

    if target_model_name is None:
        raise Exception('Target model name not found: {}.'.format(saved_dir_path))

    saved_model_path = os.path.join(
        saved_dir_path, target_model_name,
    )

    print('Loaded model: {}'.format(saved_model_path))

    return saved_model_path, target_model_name[:-4], saved_dir_path


def get_gene_symbols(data_path):
    df = pd.read_csv(data_path, index_col=0)
    data = df.loc[:, df.columns[1]:df.columns[-1]]
    gene_symbols = data.columns
    return gene_symbols


def calc_freq(data, label, threshold):
    p_values = []
    attributes = []

    n_scc = 0
    n_adeno = 0
    for i in range(data.shape[1]):
        genes = data[:, i]
        adeno  = genes[label == 1]
        scc = genes[label == 0]

        ttest = scipy.stats.ttest_ind(adeno, scc)
        p_value = ttest.pvalue
        if p_value != p_value: # if p_valus is np.nan
            p_value = 1.0
        p_values.append(p_value)

        if p_value < threshold:
            if np.sum(adeno) > np.sum(scc):  # adeno: red
                attributes.append('orangered')
                n_adeno += 1
            elif np.sum(scc) > np.sum(adeno):  # scc : blue
                attributes.append('royalblue')
                n_scc += 1
        else:
            attributes.append('silver')  # unclassified : green

    print('# of SCC: ', n_scc)
    print('# of ADC: ', n_adeno)

    return np.array(attributes), p_values


def main(config, study_name, i, n_splits,
         NUM=1000,
         CHECKPOINT_EPOCHS=[1000, 2000, 3000, 4000, 5000]):

    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    seed = check_manual_seed(config.run.seed)
    print('Using seed: {}'.format(seed))

    _, _, data_train, label_train = get_k_hold_data_loader(
        config.dataset,
        k=i,
        n_splits=n_splits,
        with_label_train=True,
    )

    attributes, p_values = calc_freq(data_train, label_train, 0.05)
    gene_symbols = get_gene_symbols(config.dataset.data_path)

    labels = {}
    for g, symbol in enumerate(gene_symbols):
        if g < NUM:
            labels[g] = symbol
        else:
            break

    attributes = attributes[: NUM]
    p_values = p_values[: NUM]

    data_train = torch.from_numpy(data_train).float().cuda(non_blocking=True)
    data_train = torch.t(data_train)

    model = get_model(config.model)
    model.cuda()
    model = nn.DataParallel(model)

    for checkpoint_epoch in CHECKPOINT_EPOCHS:
        saved_model_path, model_name, saved_dir_path = get_saved_model_path(
            config,
            study_name,
            checkpoint_epoch,
            i,
            n_splits,
        )

        model.load_state_dict(torch.load(saved_model_path)['model'])
        model.eval()

        with torch.no_grad():
            if config.model.model_name == 'MLP':
                embedding = model.module.get_embedding()

            elif config.model.model_name == 'ModifiedMLP':
                embedding = model.module.get_embedding()

            elif config.model.model_name == 'DietNetworks':
                embedding = model.module.get_embedding(data_train)

            elif config.model.model_name == 'ModifiedDietNetworks':
                embedding = model.module.get_embedding(data_train)

        embedding = embedding.detach().cpu().numpy()
        embedding = embedding[: NUM, :]

        X_tsne = TSNE(n_components=2, random_state=0).fit_transform(embedding)
        fig, ax = plt.subplots()
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10., c=attributes)

        plt.savefig(os.path.join(saved_dir_path, 'tsne_' + str(model_name) + '.png'))
        plt.savefig(os.path.join(saved_dir_path, 'tsne_' + str(model_name) + '.eps'))
        plt.clf()

        pca = PCA(n_components=2)
        X_pca = PCA(n_components=2).fit_transform(embedding)
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=10., c=attributes)

        plt.savefig(os.path.join(saved_dir_path, 'pca_' + str(model_name) + '.png'))
        plt.savefig(os.path.join(saved_dir_path, 'pca_' + str(model_name) + '.eps'))
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Mut2Vec')
    parser.add_argument('-c', '--config', help='config file',
                        default='./config/visualize.json')
    parser.add_argument('-k', '--kholds', help='number of k-holds', default=5)
    args = parser.parse_args()

    config = load_json(args.config)
    study_name = os.path.splitext(os.path.basename(args.config))[0]
    n_splits = int(args.kholds)

    # for i in range(n_splits):

    i = 0
    main(config, study_name, i, n_splits)

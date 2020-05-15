import os
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
from utils import get_output_dir_path
from utils import save_config
from utils import save_logs
from utils import save_models
from utils import load_model


def get_saved_model_path(config, checkpoint_epoch, i, n_splits):
    pattern1 = 'epoch_' + str(checkpoint_epoch)
    pattern2 = '_' + str(i) + '_' + str(n_splits) + '.pth'

    target_model_name = None
    for model_name in os.listdir(config.save.saved_dir_path):
        if model_name.startswith(pattern1):
            if model_name.endswith(pattern2):
                target_model_name = model_name
                break

    if target_model_name is None:
        raise Exception('Target model name not found.')

    saved_model_path = os.path.join(
        config.save.saved_dir_path,
        target_model_name,
    )
    return saved_model_path


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


def show_top_k_freq_symbols(gene_symbols, attributes, p_values):
    freq_symbols = pd.DataFrame({
        'gene_symbols': gene_symbols.values,
        'attributes': attributes,
        'p_values': p_values,
    })
    scc_symbols = freq_symbols[freq_symbols['attributes'] == 'royalblue']
    scc_symbols = scc_symbols.sort_values(by=['p_values'])
    print('scc_symbols: ', scc_symbols)
    adeno_symbols = freq_symbols[freq_symbols['attributes'] == 'orangered']
    adeno_symbols = adeno_symbols.sort_values(by=['p_values'])
    print('adeno_symbols: ', adeno_symbols)


def main(config, needs_save, study_name, i, n_splits, NUM=1000):
    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    seed = check_manual_seed(config.run.seed)
    print('Using seed: {}'.format(seed))

    if needs_save:
        output_dir_path = get_output_dir_path(config.save, study_name)

    train_data_loader, test_data_loader, data_train, label_train = get_k_hold_data_loader(
        config.dataset,
        k=i,
        n_splits=n_splits,
        with_label_train=True,
    )

    attributes, p_values = calc_freq(data_train, label_train, 0.05)
    gene_symbols = get_gene_symbols(config.dataset.data_path)

    # show_top_k_freq_symbols(gene_symbols, attributes, p_values)
    # input()

    labels = {}
    for i, symbol in enumerate(gene_symbols):
        if i < NUM:
            labels[i] = symbol
        else:
            break

    attributes = attributes[: NUM]
    p_values = p_values[: NUM]

    data_train = torch.from_numpy(data_train).float().cuda(non_blocking=True)
    data_train = torch.t(data_train)

    model = get_model(config.model)
    model.cuda()
    model = nn.DataParallel(model)

    # for checkpoint_epoch in config.save.checkpoint_epochs:
    for checkpoint_epoch in [500]:
        saved_model_path = get_saved_model_path(
            config,
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

        # X_tsne = TSNE(n_components=2, random_state=0).fit_transform(embedding)
        # fig, ax = plt.subplots()
        # ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10., c=attributes)
        #
        # # for i in range(NUM):
        # #     ax.annotate(labels[i], (X_tsne[i, 0], X_tsne[i, 1]))
        #
        # plt.show()
        # plt.clf()

        pca = PCA(n_components=2)
        X_pca = PCA(n_components=2).fit_transform(embedding)
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=10., c=attributes)

        for i in range(NUM):
        #     dist = np.sqrt(np.power(X_pca[i, 0], 2) + np.power(X_pca[i, 1], 2))
        #     if dist > 1.0:
            ax.annotate(labels[i], (X_pca[i, 0], X_pca[i, 1]))

        # plt.xlim([-8.0, 8.0])
        plt.ylim([-0.2, 1.0])
        plt.show()
        plt.clf()

        # pca = PCA(n_components=2)
        # pca.fit_transform(embedding)
        # axis_1= pca.components_[0]
        # print('axis_1: ', axis_1.shape)
        # print('explained_variance_ratio_: ', pca.explained_variance_ratio_)
        # print('singular_values_: ', pca.singular_values_)
        #
        # dot = np.dot(embedding, axis_1)
        # print('dot: ', dot)
        # print('dot: ', dot.shape)
        # print('argsort: ', np.argsort(dot))
        # for arg in np.argsort(dot):
        #     print(dot[arg], labels[arg])
        #     input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Mut2Vec')
    parser.add_argument('-c', '--config', help='config file',
                        default='./config/visualize.json')
    parser.add_argument('-i', '--ith-hold', default=0)
    parser.add_argument('-k', '--kholds', help='number of k-holds', default=5)
    parser.add_argument('-s', '--save', help='save logs', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)
    study_name = os.path.splitext(os.path.basename(args.config))[0]

    needs_save = args.save
    n_splits = int(args.kholds)
    i = int(args.ith_hold)

    main(config, needs_save, study_name, i, n_splits)

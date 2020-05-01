import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    for model_name in os.listdir(config.save.saved_dir_path):
        if model_name.startswith(pattern1) and model_name.endswith(pattern2):
            break

    saved_model_path = os.path.join(
        config.save.saved_dir_path,
        model_name,
    )
    return saved_model_path


def main(config, needs_save, study_name, i, n_splits):
    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    seed = check_manual_seed(config.run.seed)
    print('Using seed: {}'.format(seed))

    if needs_save:
        output_dir_path = get_output_dir_path(config.save, study_name)

    train_data_loader, test_data_loader, data_train = get_k_hold_data_loader(
        config.dataset,
        k=i,
        n_splits=n_splits,
    )
    data_train = torch.from_numpy(data_train).float().cuda(non_blocking=True)
    data_train = torch.t(data_train)

    model = get_model(config.model)
    model.cuda()
    model = nn.DataParallel(model)

    for checkpoint_epoch in config.save.checkpoint_epochs:
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
        embedding = embedding[:1000, :]

        X_tsne = TSNE(n_components=2, random_state=0).fit_transform(embedding)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5.)
        plt.show()
        plt.clf()

        X_pca = PCA(n_components=2).fit_transform(embedding)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5.)
        plt.xlim([-5.0, 5.0])
        plt.ylim([-1.0, 1.0])
        plt.show()
        plt.clf()


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

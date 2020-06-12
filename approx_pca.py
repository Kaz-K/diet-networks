import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from visualize_emb import get_saved_model_path
from dataio import get_k_hold_data_loader
from models import get_model
from utils import load_json
from utils import check_manual_seed
from utils import get_output_dir_path
from utils import save_config
from utils import save_logs
from utils import save_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_metrics(out, y):
    y_pred = torch.argmax(out, dim=1).detach().cpu().numpy()
    y_true = y.cpu().numpy()

    perfect_match = False
    if (y_true == y_pred).all():
        (TN, FP, FN, TP) = (0, 0, 0, len(y_true))
        perfect_match = True
    else:
        (TN, FP, FN, TP) = confusion_matrix(y_true, y_pred).ravel()

    return {
        'accuracy': (TP + TN) / (TP + FP + TN + FN),
    }


def main(config, needs_save, study_name, k, n_splits):
    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    seed = check_manual_seed(config.run.seed)
    print('Using seed: {}'.format(seed))

    train_data_loader, test_data_loader, data_train = get_k_hold_data_loader(
        config.dataset,
        k=k,
        n_splits=n_splits,
    )

    data_train = torch.from_numpy(data_train).float().cuda(non_blocking=True)
    data_train = torch.t(data_train)

    model = get_model(config.model)
    model.cuda()
    model = nn.DataParallel(model)

    print('count params: ', count_parameters(model.module))

    saved_model_path, _, _ = get_saved_model_path(
        config,
        study_name,
        config.model.checkpoint_epoch,
        k,
        n_splits,
    )

    model.load_state_dict(torch.load(saved_model_path)['model'])
    model.eval()

    if config.model.model_name == 'MLP':
        embedding = model.module.get_embedding()

    elif config.model.model_name == 'ModifiedMLP':
        embedding = model.module.get_embedding()

    elif config.model.model_name == 'DietNetworks':
        embedding = model.module.get_embedding(data_train)

    elif config.model.model_name == 'ModifiedDietNetworks':
        embedding = model.module.get_embedding(data_train)

    embedding = embedding.detach().cpu().numpy()

    emb_pca = PCA(n_components=2)
    emb_pca.fit_transform(embedding)

    if config.run.decomp == '1D':
        print('Approximate by 1D PCA')
        axis_1= torch.from_numpy(emb_pca.components_[0])
        score_1 = np.dot(embedding, axis_1)
        approx = np.outer(score_1, axis_1)

    elif config.run.decomp == '2D':
        print('Approximate by 2D PCA')
        axis_1= torch.from_numpy(emb_pca.components_[0])
        score_1 = np.dot(embedding, axis_1)
        axis_2= torch.from_numpy(emb_pca.components_[1])
        score_2 = np.dot(embedding, axis_2)
        approx = np.outer(score_1, axis_1) + np.outer(score_2, axis_2)
        # approx = np.outer(score_2, axis_2)

    approx = torch.from_numpy(approx).float().cuda(non_blocking=True)

    criterion = nn.CrossEntropyLoss()

    def inference(engine, batch):

        x = batch['data'].float().cuda(non_blocking=True)
        y = batch['label'].long().cuda(non_blocking=True)

        assert config.run.transposed_matrix == 'overall'
        x_t = data_train

        with torch.no_grad():
            out, _ = model.module.approx(x, approx)
            l_discriminative = criterion(out, y)
            l_total = l_discriminative

        metrics = calc_metrics(out, y)

        metrics.update({
            'l_total': l_total.item(),
            'l_discriminative': l_discriminative.item(),
        })

        torch.cuda.synchronize()

        return metrics

    evaluator = Engine(inference)

    monitoring_metrics = ['l_total', 'l_discriminative', 'accuracy']

    for metric in monitoring_metrics:
        RunningAverage(
            alpha=0.98,
            output_transform=partial(lambda x, metric: x[metric], metric=metric)
        ).attach(evaluator, metric)

    pbar = ProgressBar()
    pbar.attach(evaluator, metric_names=monitoring_metrics)

    evaluator.run(test_data_loader, 1)

    columns = ['k', 'n_splits', 'epoch', 'iteration'] + list(evaluator.state.metrics.keys())
    values = [str(k), str(n_splits), str(evaluator.state.epoch), str(evaluator.state.iteration)] \
           + [str(value) for value in evaluator.state.metrics.values()]

    values = {c: v for (c, v) in zip(columns, values)}
    values.update({
        'variance_ratio_1': emb_pca.explained_variance_ratio_[0],
        'variance_ratio_2': emb_pca.explained_variance_ratio_[1],
    })
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DietNetworks')
    parser.add_argument('-c', '--config', help='config file',
                        default='./config/config.json')
    parser.add_argument('-s', '--save', help='save logs', action='store_true')
    parser.add_argument('-k', '--kholds', help='number of k-holds', default=5)
    args = parser.parse_args()

    config = load_json(args.config)
    study_name = os.path.splitext(os.path.basename(args.config))[0]

    n_splits = int(args.kholds)

    accuracy_list = []
    variance_ratio_1 = []
    variance_ratio_2 = []

    for i in range(n_splits):
        logs = main(config,
                    args.save,
                    study_name,
                    k=i,
                    n_splits=n_splits)

        accuracy_list.append(float(logs['accuracy']))
        variance_ratio_1.append(float(logs['variance_ratio_1']))
        variance_ratio_2.append(float(logs['variance_ratio_2']))

    accuracy_list = np.array(accuracy_list)

    print('acc mean: ', np.mean(accuracy_list))
    print('acc std: ', np.std(accuracy_list))

    variance_ratio_1 = np.array(variance_ratio_1)

    print('v1 mean: ', np.mean(variance_ratio_1))
    print('v1 std: ', np.std(variance_ratio_1))

    variance_ratio_2 = np.array(variance_ratio_2)

    print('v2 mean: ', np.mean(variance_ratio_2))
    print('v2 std: ', np.std(variance_ratio_2))

    sum_variance_ratio = variance_ratio_1 + variance_ratio_2

    print('sum mean: ', np.mean(sum_variance_ratio))
    print('sum std: ', np.std(sum_variance_ratio))

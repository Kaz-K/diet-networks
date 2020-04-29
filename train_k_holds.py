import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import confusion_matrix

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

from dataio import get_k_hold_data_loader
from models import get_model
from utils import load_json
from utils import check_manual_seed
from utils import get_output_dir_path
from utils import save_config
from utils import save_logs
from utils import save_models


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

    criterion = nn.CrossEntropyLoss()

    if config.optimizer.optimizer_name == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            config.optimizer.lr,
            [0.9, 0.9999],
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError

    if needs_save:
        output_dir_path = get_output_dir_path(config.save, study_name)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    def update(engine, batch):
        model.train()

        x = batch['data'].float().cuda(non_blocking=True)
        y = batch['label'].long().cuda(non_blocking=True)

        if config.run.transposed_matrix == 'overall':
            x_t = data_train
        elif config.run.transposed_matrix == 'batch':
            x_t = torch.t(x)

        def closure():
            optimizer.zero_grad()

            if 'MLP' in config.model.model_name:
                out, x_hat = model(x)
            else:
                out, x_hat = model(x, x_t)

            l_discriminative = criterion(out, y)

            l_feature = torch.tensor(0.0).cuda()
            if config.run.w_feature_selection:
                l_feature += config.run.w_feature_selection * torch.sum(torch.abs(model.module.Ue))

            l_recon = torch.tensor(0.0).cuda()
            if config.run.w_reconstruction:
                l_recon += config.run.w_reconstruction * F.mse_loss(x, x_hat)

            l_total = l_discriminative + l_feature + l_recon

            l_total.backward()
            return l_total, l_discriminative, l_feature, l_recon, out

        l_total, l_discriminative, l_feature, l_recon, out = optimizer.step(closure)

        metrics = calc_metrics(out, y)

        metrics.update({
            'l_total': l_total.item(),
            'l_discriminative': l_discriminative.item(),
            'l_feature': l_feature.item(),
            'l_recon': l_recon.item(),
        })

        torch.cuda.synchronize()

        return metrics

    def inference(engine, batch):
        model.eval()

        x = batch['data'].float().cuda(non_blocking=True)
        y = batch['label'].long().cuda(non_blocking=True)

        if config.run.transposed_matrix == 'overall':
            x_t = data_train
        elif config.run.transposed_matrix == 'batch':
            x_t = torch.t(x)

        with torch.no_grad():
            if 'MLP' in config.model.model_name:
                out, x_hat = model(x)
            else:
                out, x_hat = model(x, x_t)

            l_discriminative = criterion(out, y)

            l_feature = torch.tensor(0.0).cuda()
            if config.run.w_feature_selection:
                l_feature += config.run.w_feature_selection * torch.sum(torch.abs(model.module.Ue))

            l_recon = torch.tensor(0.0).cuda()
            if config.run.w_reconstruction:
                l_recon += config.run.w_reconstruction * F.mse_loss(x, x_hat)

            l_total = l_discriminative + l_feature + l_recon

        metrics = calc_metrics(out, y)

        metrics.update({
            'l_total': l_total.item(),
            'l_discriminative': l_discriminative.item(),
            'l_feature': l_feature.item(),
            'l_recon': l_recon.item(),
        })

        torch.cuda.synchronize()

        return metrics

    trainer = Engine(update)
    evaluator = Engine(inference)
    timer = Timer(average=True)

    monitoring_metrics = ['l_total', 'l_discriminative', 'l_feature', 'l_recon', 'accuracy']

    for metric in monitoring_metrics:
        RunningAverage(
            alpha=0.98,
            output_transform=partial(lambda x, metric: x[metric], metric=metric)
        ).attach(trainer, metric)

    for metric in monitoring_metrics:
        RunningAverage(
            alpha=0.98,
            output_transform=partial(lambda x, metric: x[metric], metric=metric)
        ).attach(evaluator, metric)

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)
    pbar.attach(evaluator, metric_names=monitoring_metrics)

    @trainer.on(Events.STARTED)
    def events_started(engine):
        if needs_save:
            save_config(config, seed, output_dir_path)

    @trainer.on(Events.EPOCH_COMPLETED)
    def switch_training_to_evaluation(engine):
        if needs_save:
            save_logs('train', k, n_splits, trainer, trainer.state.epoch, trainer.state.iteration,
                      config, output_dir_path)

        evaluator.run(test_data_loader, max_epochs=1)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def switch_evaluation_to_training(engine):
        if needs_save:
            save_logs('val', k, n_splits, evaluator, trainer.state.epoch, trainer.state.iteration,
                      config, output_dir_path)

            if trainer.state.epoch in [100, 200, 300, 400, 500]:
                save_models(model, optimizer, k, n_splits, trainer.state.epoch, trainer.state.iteration,
                            config, output_dir_path)

        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    @evaluator.on(Events.EPOCH_COMPLETED)
    def show_logs(engine):
        columns = ['k', 'n_splits', 'epoch', 'iteration'] + list(engine.state.metrics.keys())
        values = [str(k), str(n_splits), str(engine.state.epoch), str(engine.state.iteration)] \
               + [str(value) for value in engine.state.metrics.values()]

        message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                              max_epoch=config.run.n_epochs,
                                                              i=engine.state.iteration,
                                                              max_i=len(train_data_loader))

        for name, value in zip(columns, values):
            message += ' | {name}: {value}'.format(name=name, value=value)

        pbar.log_message(message)

    timer.attach(trainer,
                 start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.run.n_epochs, config.run.n_epochs * len(train_data_loader))
    )

    trainer.run(train_data_loader, config.run.n_epochs)


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
    for i in range(n_splits):
        main(config, args.save, study_name, k=i, n_splits=n_splits)

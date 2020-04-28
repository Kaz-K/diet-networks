import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from dataio import get_data_loader
from models import get_model
from utils import load_json
from utils import check_manual_seed
from utils import adjust_learning_rate
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


def main(config, needs_save, study_name):
    if config.run.visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices

    seed = check_manual_seed(config.run.seed)
    print('Using seed: {}'.format(seed))

    train_data_loader, test_data_loader, data_train = get_data_loader(config.dataset)
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

    def update(engine, batch):
        adjust_learning_rate(
            optimizer,
            init_lr=config.optimizer.lr,
            epoch=engine.state.epoch,
            interval_epochs=config.optimizer.interval_epochs,
        )

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

            loss = criterion(out, y)
            loss.backward()
            return loss, out

        loss, out = optimizer.step(closure)

        metrics = calc_metrics(out, y)
        metrics.update({
            'loss': loss.item(),
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

            loss = criterion(out, y)

        metrics = calc_metrics(out, y)
        metrics.update({
            'loss': loss.item(),
        })

        torch.cuda.synchronize()

        return metrics

    trainer = Engine(update)
    evaluator = Engine(inference)
    timer = Timer(average=True)

    monitoring_metrics = ['loss', 'accuracy']

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

    @trainer.on(Events.EPOCH_COMPLETED)
    def switch_training_to_evaluation(engine):
        evaluator.run(test_data_loader, max_epochs=1)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def show_logs(engine):
        columns = ['epoch', 'iteration'] + list(engine.state.metrics.keys())
        values = [str(engine.state.epoch), str(engine.state.iteration)] \
               + [str(value) for value in engine.state.metrics.values()]

        message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                              max_epoch=config.run.n_epochs,
                                                              i=engine.state.iteration,
                                                              max_i=len(train_data_loader))

        for name, value in zip(columns, values):
            message += ' | {name}: {value}'.format(name=name, value=value)

        pbar.log_message(message)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(
            engine.state.epoch, timer.value())
        )
        timer.reset()

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
    args = parser.parse_args()

    config = load_json(args.config)
    study_name = os.path.splitext(os.path.basename(args.config))[0]

    main(config, args.save, study_name)

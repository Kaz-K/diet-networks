import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import torch


def save_config(config, seed=None, output_dir_path=None):
    config_to_save = defaultdict(dict)
    for key, child in config._asdict().items():
        for k, v in child._asdict().items():
            config_to_save[key][k] = v

    if seed:
        config_to_save['seed'] = seed
    if output_dir_path:
        config_to_save['output_dir_path'] = output_dir_path

    save_path = os.path.join(output_dir_path, 'config.json')
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f)


def save_models(model, optimizer, epoch, iteration, config, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)

    path = os.path.join(
        output_dir_path, 'epoch_{}_iteration_{}.pth'.format(epoch, iteration)
    )

    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
    }, path)


def save_logs(mode, k, n_splits, engine, epoch, iteration, config, output_dir_path):
    if mode == 'train':
        fname = os.path.join(output_dir_path, 'train_logs_' + str(k) + '.csv')
    elif mode == 'val':
        fname = os.path.join(output_dir_path, 'val_logs_' + str(k) + '.csv')
    else:
        raise NotImplementedError

    columns = ['k', 'n_splits', 'epoch', 'iteration'] + list(engine.state.metrics.keys())
    values = [str(k), str(n_splits), str(epoch), str(iteration)] \
           + [str(value) for value in engine.state.metrics.values()]

    with open(fname, 'a') as f:
        if f.tell() == 0:
            print(','.join(columns), file=f)
        print(','.join(values), file=f)

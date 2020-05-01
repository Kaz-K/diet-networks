import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils import data

from .dataset import LungCancerDataset


def get_data_loader(data_config):
    df = pd.read_csv(data_config.data_path, index_col=0)
    data_frame = df.loc[:, df.columns[1]:df.columns[-1]]
    data_frame = data_frame.values.astype(np.int32)

    if data_config.as_binary:
        data_frame[data_frame >= 1] = 1
        data_frame[data_frame < 1] = 0

    label = df['ONCOTREE_CODE'].values
    label[label == 'LUAD'] = 1
    label[label == 'LUSC'] = 0
    label = label.astype(np.int32)

    data_train, data_test, label_train, label_test = train_test_split(
        data_frame, label,
        test_size=data_config.test_size,
        random_state=data_config.random_state
    )

    train_dataset = LungCancerDataset(data_train, label_train)
    test_dataset = LungCancerDataset(data_test, label_test)

    train_data_loder = data.DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        drop_last=True,
    )

    test_data_loader = data.DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        drop_last=True,
    )

    return train_data_loder, test_data_loader, data_train


def get_k_hold_data_loader(data_config, k, n_splits):
    df = pd.read_csv(data_config.data_path, index_col=0)
    data_frame = df.loc[:, df.columns[1]:df.columns[-1]]
    data_frame = data_frame.values.astype(np.int32)
    data_frame = data_frame[:data_config.data_size, :]

    if data_config.as_binary:
        data_frame[data_frame >= 1] = 1
        data_frame[data_frame < 1] = 0

    label = df['ONCOTREE_CODE'].values
    label[label == 'LUAD'] = 1
    label[label == 'LUSC'] = 0
    label = label.astype(np.int32)
    label = label[:data_config.data_size]

    kf = KFold(n_splits=n_splits)
    train_index, test_index = list(kf.split(data_frame, label))[k]

    data_train = data_frame[train_index, :]
    data_test = data_frame[test_index, :]
    label_train = label[train_index]
    label_test = label[test_index]

    train_dataset = LungCancerDataset(data_train, label_train)
    test_dataset = LungCancerDataset(data_test, label_test)

    train_data_loder = data.DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        drop_last=True,
    )

    test_data_loader = data.DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        drop_last=True,
    )

    return train_data_loder, test_data_loader, data_train


def get_k_hold_data_table(data_config, k, n_splits):
    df = pd.read_csv(data_config.data_path, index_col=0)
    data_frame = df.loc[:, df.columns[1]:df.columns[-1]]
    data_frame = data_frame.values.astype(np.int32)
    data_frame = data_frame[:data_config.data_size, :]

    if data_config.as_binary:
        data_frame[data_frame >= 1] = 1
        data_frame[data_frame < 1] = 0

    label = df['ONCOTREE_CODE'].values
    label[label == 'LUAD'] = 1
    label[label == 'LUSC'] = 0
    label = label.astype(np.int32)
    label = label[:data_config.data_size]

    kf = KFold(n_splits=n_splits)
    train_index, test_index = list(kf.split(data_frame, label))[k]

    data_train = data_frame[train_index, :]
    data_test = data_frame[test_index, :]
    label_train = label[train_index]
    label_test = label[test_index]
    
    return data_train, label_train, data_test, label_test

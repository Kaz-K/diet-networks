import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch.utils import data


def get_data_loader(data_config):
    df = pd.read_csv(data_config.data_path, index_col=0)
    data = df.loc[:, df.columns[1]:df.columns[-1]]
    data = data.values.astype(np.int32)

    if data_config.as_binary:
        data[data >= 1] = 1
        data[data < 1] = 0

    label = df['ONCOTREE_CODE'].values
    label[label == 'LUAD'] = 1
    label[label == 'LUSC'] = 0
    label = label.astype(np.int32)

    data_train, data_test, label_train, label_test = train_test_split(
        data, label,
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
    )

    test_data_loader = data.DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    return train_data_loder, test_data_loader

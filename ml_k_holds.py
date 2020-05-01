import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd

from dataio import get_k_hold_data_table
from models import get_model
from utils import load_json
from utils import check_manual_seed
from utils import get_output_dir_path
from utils import save_config
from utils import save_logs
from utils import save_models


def calc_metrics(y_pred, y_true):
    (TN, FP, FN, TP) = confusion_matrix(y_true, y_pred).ravel()

    return {
        'accuracy': (TP + TN) / (TP + FP + TN + FN),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DietNetworks')
    parser.add_argument('-c', '--config', help='config file',
                        default='./config/config.json')
    parser.add_argument('-s', '--save', help='save logs', action='store_true')
    parser.add_argument('-k', '--kholds', help='number of k-holds', default=5)
    args = parser.parse_args()

    config = load_json(args.config)
    study_name = os.path.splitext(os.path.basename(args.config))[0]

    if args.save:
        output_dir_path = get_output_dir_path(config.save, study_name)
    else:
        output_dir_path = None

    n_splits = int(args.kholds)


    accuracy_list = []
    for i in tqdm(range(n_splits)):
        data_train, label_train, data_test, label_test = get_k_hold_data_table(
            config.dataset,
            k=i,
            n_splits=n_splits,
        )

        # lr = LogisticRegression()
        # lr.fit(data_train, label_train)
        # pred = lr.predict(data_test)

        # clf = svm.SVC()
        # clf.fit(data_train, label_train)
        # pred = clf.predict(data_test)

        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(data_train, label_train)
        pred = clf.predict(data_test)

        metrics = calc_metrics(pred, label_test)

        accuracy_list.append(metrics['accuracy'])

    print('average: ', np.average(accuracy_list))
    print('std: ', np.std(accuracy_list))

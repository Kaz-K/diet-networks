import os
import numpy as np
import pandas as pd


MODE = 'val'
SAVE_ROOT_DIR = './result_comp'
LOWER_EPOCH = 4990
UPPER_EPOCH = 5000
N_SPLITS = 5


if __name__ == '__main__':
    for dir_name in os.listdir(SAVE_ROOT_DIR):
        dir_path = os.path.join(SAVE_ROOT_DIR, dir_name)

        result = None
        for i in range(N_SPLITS):
            log_file_path = os.path.join(
                dir_path, MODE + '_logs_' + str(i) + '.csv'
            )

            df = pd.read_csv(log_file_path, header=0)['accuracy'][LOWER_EPOCH:UPPER_EPOCH].to_frame()
            df = df.rename(columns={'accuracy': 'accuracy_' + str(i)})

            if result is None:
                result = df
            else:
                result = pd.concat([result, df], axis=1)

        result = result.values

        mean = np.mean(result)
        std = np.std(result)

        print(dir_name)
        print('  mean: ', mean)
        print('  std: ', std)

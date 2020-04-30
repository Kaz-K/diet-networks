import os
import numpy as np
import pandas as pd


studies = ['diet_baseline_Apr', 'diet_fs_0.0001_Apr', 'diet_fs_0.001_Apr', 'diet_fs_Apr',
           'diet_recon_0.001_Apr', 'diet_recon_0.01_Apr',
           'mlp_baseline_Apr', 'mlp_fs_0.0001_Apr', 'mlp_fs_0.001_Apr', 'mlp_fs_Apr',
           'mlp_recon_0.001_Apr', 'mlp_recon_0.01_Apr']

modes = ['val']

result_root_dir = './result'

lower_epoch = 400
upper_epoch = 500
n_splits = 5


if __name__ == '__main__':
    dir_names = os.listdir(result_root_dir)

    for study in studies:
        for dir_name in dir_names:
            if study in dir_name:
                study_dir_path = os.path.join(result_root_dir, dir_name)
                break

        result = None
        for mode in modes:
            for i in range(n_splits):
                log_file_path = os.path.join(
                    study_dir_path, mode + '_logs_' + str(i) + '.csv'
                )

                df = pd.read_csv(log_file_path, header=0)['accuracy'][lower_epoch:upper_epoch].to_frame()
                df = df.rename(columns={'accuracy': 'accuracy_' + str(i)})

                if result is None:
                    result = df
                else:
                    result = pd.concat([result, df], axis=1)

            result = result.values

            mean = np.mean(result)
            std = np.std(result)

            print(study)
            print('mean: ', mean)
            print('std: ', std)

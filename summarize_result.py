import os
import numpy as np
import pandas as pd

studies = [
    'diet_baseline_May20_00-53-31',
    'diet_recon_0.1_May20_03-21-16',
    'diet_recon_0.01_May20_03-04-50',
    'diet_recon_0.001_May20_02-48-22',
    'diet_fs_May20_02-32-12',
    'diet_fs_recon_0.1_May20_02-15-28',
    'diet_fs_recon_0.01_May20_01-58-46',
    'diet_fs_recon_0.001_May20_01-42-06',
    'diet_fs_0.001_May20_01-25-58',
    'diet_fs_0.0001_May20_01-09-44',
    'mlp_baseline_May20_03-37-40',
    'mlp_fs_May20_04-26-52',
    'mlp_fs_0.001_May20_04-09-57',
    'mlp_fs_0.0001_May20_03-53-22',
]

modes = ['val']
result_root_dir = './result_fixed'
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
            print('  mean: ', mean)
            print('  std: ', std)

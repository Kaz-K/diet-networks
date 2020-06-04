import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


result_root_dir = './result_fixed2'
lower_epoch = 400
upper_epoch = 500
n_splits = 5
n_epochs = 500


if __name__ == '__main__':
    for dir_name in os.listdir(result_root_dir):
        output_dir_path = os.path.join(result_root_dir, dir_name)
        print(output_dir_path)

        for mode in ['train', 'val']:
            result = None
            for i in range(n_splits):
                log_file_path = os.path.join(
                    output_dir_path, mode + '_logs_' + str(i) + '.csv'
                )
                df = pd.read_csv(log_file_path, header=0)['accuracy'].to_frame()
                df = df.rename(columns={'accuracy': 'accuracy_' + str(i)})

                if result is None:
                    result = df
                else:
                    result = pd.concat([result, df], axis=1)

            result.plot()
            plt.title(os.path.basename(output_dir_path))
            plt.ylim([0, 1.1])
            plt.xlim([0, n_epochs])
            plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)
            plt.savefig(os.path.join(output_dir_path, dir_name + '_' + mode + '_result.eps'))
            plt.savefig(os.path.join(output_dir_path, dir_name + '_' + mode + '_result.png'))
            plt.clf()

            mean = result.mean(axis=1)
            std = result.std(axis=1)
            plt.figure()
            plt.title(os.path.basename(output_dir_path))
            plt.ylim([0, 1.5])
            plt.xlim([0, n_epochs])
            plt.plot(range(0, n_epochs), mean, 'b')
            plt.fill_between(range(0, n_epochs), mean - 2 * std, mean + 2 * std, color='b', alpha=0.2)
            plt.savefig(os.path.join(output_dir_path, dir_name + '_' + mode + '_mean.eps'))
            plt.savefig(os.path.join(output_dir_path, dir_name + '_' + mode + '_mean.png'))
            plt.clf()

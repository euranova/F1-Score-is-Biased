""" Produces the results presented in Table 1 in the paper. """

import os
import re

import numpy as np

import click
from tqdm import tqdm
from sklearn.svm import OneClassSVM

from utils.datasets import load_dataset, _DATASET_FILES
from utils.helpers import get_subset
from utils.protocols import algo1, algo2


@click.command()
@click.option('--output_file', default='./results/table1.csv', help='Output file')
@click.option('--n_runs', default=100, help='Number of runs to do', type=int)
@click.option('-d', '--datasets', default=["thyroid", "arrhythmia"], multiple=True,
              type=click.Choice(list(_DATASET_FILES), case_sensitive=False),
              help='Datasets to use. Multiple datasets can be entered by reentering the option.')
@click.option('-s', '--settings', multiple=True, type=str, default=["1.2e", "2.2e", "2.05e", "2.05o"],
              help=r"Settings to use, following the format '[12]\.\d+[oe]'. "
                   "For example, 1.25e stands for algo 1, .25 test_size and estimated threshold. "
                   "2.5o stands for algo 2, .5 test_size and optimal threshold. "
                   "Multiple settings can be entered by reentering the option.")
def main(output_file, n_runs, datasets, settings):
    for setting in settings:
        if not re.match(r"^[12]\.\d+[oe]$", setting):
            raise ValueError(f"setting can't be {setting}, it has to be of the form '[12]\\.\\d+[oe]'")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    print(f"[INFO] Output file: {output_file}")

    with open(output_file, 'w') as file:
        file.write("dataset_name,algo,test_size,thresholds,f1_mean,f1_std,auc_mean,auc_std,avpr_mean,avpr_std\n")

        for setting in settings:
            algo = algo1 if setting.startswith('1') else algo2
            test_size = float(setting[1:-1])
            threshold = "estimated" if setting.endswith('e') else "optimal"
            algo_name = algo.__name__
            for dataset_name in datasets:
                print(f"Dataset {dataset_name}, algo {algo_name}, test_size {test_size}, threshold {threshold}")

                x, y = load_dataset(dataset_name)

                f1_scores = []
                aucs = []
                avprs = []

                for _ in tqdm(range(n_runs)):
                    clf = OneClassSVM(gamma="auto", nu=0.9)

                    x_run, y_run = get_subset(x, y) if dataset_name == "kddcup" else (x.copy(), y.copy())
                    f1_score, auc, avpr, optimal_f1_score = algo(x_run, y_run, test_size, clf)

                    f1_scores.append(f1_score if threshold == "estimated" else optimal_f1_score)
                    aucs.append(auc)
                    avprs.append(avpr)

                f1_mean = np.mean(f1_scores)
                f1_std = np.std(f1_scores, ddof=1)

                auc_mean = np.mean(aucs)
                auc_std = np.std(aucs, ddof=1)

                avpr_mean = np.mean(avprs)
                avpr_std = np.std(avprs, ddof=1)

                print(f"{dataset_name}/{algo_name}/{test_size}/{threshold} -> "
                      f"F1: {f1_mean} ({f1_std}); AUC: {auc_mean} ({auc_std}); AVPR: {avpr_mean} ({avpr_std})")

                file.write(f"{dataset_name},{algo_name},{test_size},{threshold},"
                           f"{f1_mean},{f1_std},{auc_mean},{auc_std},{avpr_mean},{avpr_std}\n")
                file.flush()


if __name__ == "__main__":
    main()

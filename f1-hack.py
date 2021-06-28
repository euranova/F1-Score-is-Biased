import os
import numpy as np

import click

from tqdm import tqdm
from sklearn.svm import OneClassSVM

from utils.datasets import load_dataset
from utils.protocols import algo1, algo2, get_subset


@click.command()
@click.option('--output_file', default='./results/table1.csv', help='Output file')
@click.option('--n_runs', default=100, help='number of run to do')
def main(output_file, n_runs):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    print(f"[INFO] Output file: {output_file}")

    datasets = ["arrhythmia", "thyroid"] #, "kddcup"]

    algorithms = [algo1, algo2, algo2, algo2]
    test_sizes = [0.2, 0.2, 0.05, 0.05]
    thresholds = ["estimated", "estimated", "estimated", "optimal"]

    with open(output_file, 'w') as file:
        file.write(f"dataset_name,algo,test_size,thresholds,f1_mean,f1_std,auc_mean,auc_std,avpr_mean,avpr_std\n")

        for algo, test_size, threshold in zip(algorithms, test_sizes, thresholds):
            algo_name = algo.__name__
            for dataset_name in datasets:
                print(f"Dataset {dataset_name}, algo {algo_name}, test_size {test_size}, threshold {threshold}")

                try:
                    x, y = load_dataset(dataset_name)

                    f1_scores = []
                    auc_scores = []
                    avpr_scores = []
                    optimal_f1_scores = []

                    for _ in tqdm(range(n_runs)):
                        clf = OneClassSVM(gamma="auto", nu=0.9)

                        x_run, y_run = get_subset(x, y) if dataset_name == "kddcup" else (x.copy(), y.copy())
                        f1_score, auc_score, avpr_score, f1_optimal_score = algo(x_run, y_run, test_size, clf)

                        f1_scores.append(f1_score)
                        auc_scores.append(auc_score)
                        avpr_scores.append(avpr_score)
                        optimal_f1_scores.append(f1_optimal_score)

                    f1_scores = f1_scores if threshold == "estimated" else optimal_f1_scores

                    f1_mean = np.mean(f1_scores)
                    f1_std = np.std(f1_scores)

                    auc_mean = np.mean(auc_scores)
                    auc_std = np.std(auc_scores)

                    avpr_mean = np.mean(avpr_scores)
                    avpr_std = np.std(avpr_scores)

                    print(f"{dataset_name}/{algo_name}/{test_size}/{threshold} -> "
                          f"F1: {f1_mean} ({f1_std}); AUC: {auc_mean} ({auc_std}); AVPR: {avpr_mean} ({avpr_std})")

                    file.write(f"{dataset_name},{algo_name},{test_size},{threshold},"
                               f"{f1_mean},{f1_std},{auc_mean},{auc_std},{avpr_mean},{avpr_std}\n")
                    file.flush()

                except Exception as inst:
                    print(inst)
                    continue


if __name__ == "__main__":
    main()

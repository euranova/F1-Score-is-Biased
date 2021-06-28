import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support as prf


def get_scores(labels, predictions):
    # compute precision, recall, f1_score on positive class (= anomaly)
    precision, recall, f_score, _ = prf(labels, predictions, average='binary', zero_division=1)
    return precision, recall, f_score


def get_auc_score(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def get_avpr_score(labels, scores):
    return average_precision_score(labels, scores)


def compute_f1_score(tp, fp, fn):
    return (2*tp) / (2*tp + fp + fn)


def get_optimal_f1_score(labels, scores):
    # Sort scores lower to bigger
    indexes = np.argsort(scores)

    tp = np.sum(labels)
    fp = len(labels) - tp
    fn = 0
    tn = 0

    optimal_f1_score = compute_f1_score(tp, fp, fn)
    for label in labels[indexes]:
        if label == 1:
            tp -= 1
            fn += 1
        else:
            fp -= 1
            tn += 1

        optimal_f1_score = max(compute_f1_score(tp, fp, fn), optimal_f1_score)
    return optimal_f1_score
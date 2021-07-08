""" Contains metric-related functions. """

import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support as prf


def get_precision_recall_f1score(labels, predictions):
    """ Computes the precision, recall and F1-score using anomalies as the positive class.

    :param labels: np.ndarray(np.uint8); labels of the samples
    :param predictions: np.ndarray(Float); corresponding binary predictions
    :return: (Float, Float, Float); precision, recall and F1-score
    """
    if all(labels == labels[0]):
        return None, None, None
    precision, recall, f1_score, _ = prf(labels, predictions, average='binary', zero_division=1)
    return precision, recall, f1_score


def get_auc(labels, scores):
    """ Computes the auc.

    :param labels: np.ndarray(np.uint8); labels of the samples
    :param scores: np.ndarray(Float); corresponding anomaly scores
    :return: Float; auc
    """
    if all(labels == labels[0]):
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def get_avpr(labels, scores):
    """ Computes the avpr using anomalies as the positive class.

    :param labels: np.ndarray(np.uint8); labels of the samples
    :param scores: np.ndarray(Float); corresponding anomaly scores
    :return: Float; avpr
    """
    if all(labels == labels[0]):
        return None
    return average_precision_score(labels, scores)


def compute_f1_score(tp, fp, fn):
    """ Computes the F1-score.

    :param tp: Integer: number of true positives
    :param fp: Integer: number of false positives
    :param fn: Integer: number of false negatives
    :return: Float: F1-score
    """
    return (2*tp) / (2*tp + fp + fn)


def get_optimal_f1_score(labels, scores):
    """ Computes the optimal-threshold F1-score using anomalies as the positive class.

    :param labels: np.ndarray(np.uint8); labels of the samples
    :param scores: np.ndarray(Float); corresponding anomaly scores
    :return: Float: optimal-threshold F1-score
    """
    if all(labels == labels[0]):
        return None
    # Sort scores lower to bigger
    indexes = np.argsort(scores)

    tp = np.sum(labels)  # every sample starts being predicted as anomalous
    fp = len(labels) - tp
    fn = 0
    tn = 0

    optimal_f1_score = compute_f1_score(tp, fp, fn)
    for label in labels[indexes]:
        if label == 1:  # an anomalous sample is predicted normal
            tp -= 1
            fn += 1
        else:  # a normal sample is predicted normal
            fp -= 1
            tn += 1

        optimal_f1_score = max(compute_f1_score(tp, fp, fn), optimal_f1_score)
    return optimal_f1_score

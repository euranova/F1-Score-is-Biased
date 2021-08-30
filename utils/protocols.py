""" Contains the two algorithms presented in the paper as well as sub-functions used in these. """

import numpy as np
from sklearn.model_selection import train_test_split

from .datasets import normalize_data
from .helpers import get_model_scores, get_threshold
from .scores import get_precision_recall_f1score, get_auc, get_avpr, get_optimal_f1_score


def algo1(samples, labels, test_size, model, compute_metrics=True):
    """ Algorithm 1 as described in the paper.

    :param samples: np.ndarray(Float); samples to use to train and test the given model
    :param labels: np.ndarray(np.uint8); corresponding labels (1 is anomalous)
    :param test_size: Float; proportion of samples to use in the test set
    :param model: sklearn model; model to train and evaluate
    :param compute_metrics: Bool; whether to compute the metrics or not
    :return:
        if compute_metrics is True: (Float, Float, Float, Float); F1-score, AUC, AVPR and optimal-threshold F1-score
        if compute_metrics is False: (np.ndarray(Float), np.ndarray(Float), np.ndarray(Float)):
            labels, scores and binary predictions of the test set
    """
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size)

    # Clean the train set (y_train: 1 for anomaly, 0 for normal)
    x_clean = x_train[y_train == 0]

    # Normalize the samples fitting on x_clean
    x_clean, x_train, x_test = normalize_data(fit_on=x_clean, transform=(x_clean, x_train, x_test))

    # Train the model using x_train
    model.fit(x_clean)

    # Compute the scores obtained on the train set
    scores_on_train_set = get_model_scores(model, x_train)

    # Estimate the contamination rate based on the train set (in [0, 1])
    contamination_rate = np.sum(y_train) / len(y_train)

    # Compute a threshold based on the train set
    threshold = get_threshold(scores_on_train_set, contamination_rate)

    # Compute the scores obtained on the test set
    scores_on_test_set = get_model_scores(model, x_test)

    # Compute the binary predictions
    y_test_predicted = (scores_on_test_set >= threshold).astype(int)

    # Return labels, scores and predictions in case we want to explore other metrics
    if not compute_metrics:
        return y_test, scores_on_test_set, y_test_predicted

    # Get all metric values
    _, _, f1_score = get_precision_recall_f1score(y_test, y_test_predicted)
    auc = get_auc(y_test, scores_on_test_set)
    avpr = get_avpr(y_test, scores_on_test_set)
    optimal_f1__score = get_optimal_f1_score(y_test, scores_on_test_set)

    return f1_score, auc, avpr, optimal_f1__score


def algo2(samples, labels, test_size, model, compute_metrics=True):
    """ Algorithm 2 as described in the paper.

    :param samples: np.ndarray(Float); samples to use to train and test the given model
    :param labels: np.ndarray(np.uint8); corresponding labels (1 is anomalous)
    :param test_size: Float; proportion of samples to use in the test set
    :param model: sklearn model; model to train and evaluate
    :param compute_metrics: Bool; whether to compute the metrics or not
    :return:
        if compute_metrics is True: (Float, Float, Float, Float); F1-score, AUC, AVPR and optimal-threshold F1-score
        if compute_metrics is False: (np.ndarray(Float), np.ndarray(Float), np.ndarray(Float)):
            labels, scores and binary predictions of the test set
    """
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size, stratify=labels)

    # Clean the train set (y_train: 1 for anomaly, 0 for normal)
    x_clean = x_train[y_train == 0]

    # Re-inject anomalies in the test set
    x_test = np.r_[x_test, x_train[y_train == 1]]
    y_test = np.r_[y_test, y_train[y_train == 1]]

    # Normalize the samples fitting on x_clean
    x_clean, x_test = normalize_data(fit_on=x_clean, transform=(x_clean, x_test))

    # Train the model using x_train
    model.fit(x_clean)

    # Compute the scores obtained on the test set
    scores_on_test_set = get_model_scores(model, x_test)

    # Compute the contamination rate of the test set (in [0, 1])
    contamination_rate = np.sum(y_test) / len(y_test)

    # Compute a threshold based on the test set
    threshold = get_threshold(scores_on_test_set, contamination_rate)

    # Compute the binary predictions
    y_test_predicted = (scores_on_test_set >= threshold).astype(int)

    # Return labels, scores and predictions in case we want to explore other metrics
    if not compute_metrics:
        return y_test, scores_on_test_set, y_test_predicted

    # Get all metric values
    _, _, f1_score = get_precision_recall_f1score(y_test, y_test_predicted)
    auc = get_auc(y_test, scores_on_test_set)
    avpr = get_avpr(y_test, scores_on_test_set)
    optimal_f1__score = get_optimal_f1_score(y_test, scores_on_test_set)

    return f1_score, auc, avpr, optimal_f1__score


def algo2_end(x_train, x_test, y_test, model):
    """ Algorithm 2 as described in the paper, but the splitting is already done and there is no normalisation step.

    :param x_train: np.ndarray(Float); samples to use to train the given model
    :param x_test: np.ndarray(Float); samples to use to test the given model
    :param y_test: np.ndarray(np.uint8); test labels (1 is anomalous)
    :param model: sklearn model; model to train and evaluate
    :return: (Float, Float, Float, Float); F1-score, AUC, AVPR and optimal-threshold F1-score
    """
    model.fit(x_train)
    scores_on_test_set = get_model_scores(model, x_test)
    contamination_rate = np.sum(y_test) / len(y_test)
    threshold = get_threshold(scores_on_test_set, contamination_rate)
    y_test_predicted = (scores_on_test_set >= threshold).astype(int)

    _, _, f1_score = get_precision_recall_f1score(y_test, y_test_predicted)
    auc = get_auc(y_test, scores_on_test_set)
    avpr = get_avpr(y_test, scores_on_test_set)
    optimal_f1__score = get_optimal_f1_score(y_test, scores_on_test_set)

    return f1_score, auc, avpr, optimal_f1__score

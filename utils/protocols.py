import numpy as np
from sklearn.model_selection import train_test_split

from .datasets import normalize_data
from .scores import get_scores, get_auc_score, get_avpr_score, get_optimal_f1_score


def get_model_scores(model, data):
    # Get the model prediction (lower value for anomalous data)
    scores = model.score_samples(data)
    # we want a higher score for anomalous data
    scores *= -1
    return scores


def get_threshold(scores, contamination_rate_estimation):
    return np.percentile(scores, (1-contamination_rate_estimation)*100)


def get_subset(x, y, subset=0.1):
    indexes = np.random.choice(len(x), int(len(x) * subset), replace=False)
    return x[indexes].copy(), y[indexes].copy()


def algo1(x, y, test_size, model, compute_metrics=True):
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # Clean train set (y_train: 1 for anomaly, 0 for normal)
    x_clean = x_train[y_train == 0]

    # Normalize the data fitting on x_clean
    x_clean, x_train, x_test = normalize_data(fit_on=x_clean, transform=(x_clean, x_train, x_test))

    # Train the model using x_train
    model.fit(x_clean)

    # Compute score on train set
    s_train = get_model_scores(model, x_train)

    # Compute contamination rate on train set (in [0, 1])
    cont = np.sum(y_train) / len(y_train)

    # Compute threshold with train set
    thresh = get_threshold(s_train, cont)

    # Compute estimation of test set
    s_test = get_model_scores(model, x_test)

    # Compute different scores
    y_hat_test = (s_test >= thresh).astype(int)

    # Return labels, score, predictions if we want to explore other metrics
    if not compute_metrics:
        return y_test, s_test, y_hat_test

    # Get all scores
    _, _, f1_score = get_scores(y_test, y_hat_test)
    auc_score = get_auc_score(y_test, s_test)
    avpr_score = get_avpr_score(y_test, s_test)
    f1_optimal_score = get_optimal_f1_score(y_test, s_test)

    return f1_score, auc_score, avpr_score, f1_optimal_score


def algo2(x, y, test_size, model):
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # Clean train set
    x_clean = x_train[y_train == 0]

    # Re-inject anomalous data in test set
    x_test = np.concatenate((x_test, x_train[y_train == 1]), axis=0)
    y_test = np.concatenate((y_test, y_train[y_train == 1]), axis=0)

    # Normalize data
    x_clean, x_test = normalize_data(fit_on=x_clean, transform=(x_clean, x_test))

    # Train the model using x_train
    model.fit(x_clean)

    # Compute estimation of test set
    s_test = get_model_scores(model, x_test)

    # Compute contamination rate on test set (in [0, 1])
    cont = np.sum(y_test) / len(y_test)

    # Compute threshold with test set
    thresh = get_threshold(s_test, cont)

    # Compute different scores
    y_hat_test = (s_test >= thresh).astype(int)

    # Get all scores
    _, _, f1_score = get_scores(y_test, y_hat_test)
    auc_score = get_auc_score(y_test, s_test)
    avpr_score = get_avpr_score(y_test, s_test)
    f1_optimal_score = get_optimal_f1_score(y_test, s_test)

    return f1_score, auc_score, avpr_score, f1_optimal_score

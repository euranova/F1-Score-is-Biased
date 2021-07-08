""" Contains helpers that cannot go in other categories. """

import numpy as np


def get_model_scores(model, samples):
    """ Computes the score given by the model to given samples.

    :param model: sklearn model; model to use for anomaly detection
    :param samples: np.ndarray(Float); samples to make a prediction for
    :return: np.ndarray(Float); scores (higher is more anomalous)
    """
    # Get the model prediction (lower value for anomalous data)
    scores = model.score_samples(samples)
    # we want a higher score for anomalous data
    return -scores


def get_threshold(scores, contamination_rate_estimation):
    """ Computes the fp = fn threshold.

    :param scores: np.ndarray(Float); anomaly-scores to base the threshold on
    :param contamination_rate_estimation: Float; estimation of the contamination rate
    :return: Float; fp = fn threshold
    """
    return np.percentile(scores, (1-contamination_rate_estimation)*100)


def get_subset(samples, labels, subset=0.1):
    """ Selects without replacement samples from the given sets.

    :param samples: np.ndarray(Float); samples to select from
    :param labels: np.ndarray(np.uint8); corresponding labels
    :param subset: Float; proportion of samples to select
    :return: (np.ndarray(Float), np.ndarray(np.uint8)); subset of samples and corresponding labels.
    """
    indexes = np.random.choice(len(samples), int(len(samples) * subset), replace=False)
    return samples[indexes].copy(), labels[indexes].copy()


def cont_to_anomalies_per_clean_sample(contamination_rate):
    """ Computes the number of anomalies per clean sample needed to obtain the expected contamination rate.

    :param contamination_rate: np.ndarray(Float) or Float in [0, 1[; contamination rate expected
    :return: np.ndarray(Float) or Float; number of anomalies per clean samples needed
    """
    return contamination_rate / (1 - contamination_rate)

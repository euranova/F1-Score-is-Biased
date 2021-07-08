""" Contains dataset-related functions. """

import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.io
import mat73

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import cont_to_anomalies_per_clean_sample

_DATASET_URLS = {
    "kddcup": "https://datahub.io/machine-learning/kddcup99/r/kddcup99.csv",
    "arrhythmia": "https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1",
    "thyroid": "https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1"
}

_DATASET_FILES = {
    "kddcup": "kddcup99_csv.csv",
    "arrhythmia": "arrhythmia.mat",
    "thyroid": "thyroid.mat"
}

_DATASET_FOLDER = os.path.abspath(os.path.join(__file__, "../../datasets/"))

if not os.path.exists(_DATASET_FOLDER):
    os.makedirs(_DATASET_FOLDER)


def _get_path(dataset_name):
    """ Returns the path to the required dataset, downloading the dataset if necessary.

    :param dataset_name: String (in _DATASET_FILES); dataset required
    :return: String; path to the dataset
    """
    dataset_path = os.path.join(_DATASET_FOLDER, _DATASET_FILES[dataset_name])
    if not os.path.isfile(dataset_path):
        print(f"[DOWNLOAD] Download dataset {dataset_name} from {_DATASET_URLS[dataset_name]}")
        with urllib.request.urlopen(_DATASET_URLS[dataset_name]) as u, open(dataset_path, "wb") as f:
            f.write(u.read())
    return dataset_path


def _load_kddcup():
    """ Loads the kddcup dataset.

    :return: (np.ndarray(Float), np.ndarray(np.uint8)); ordinal-encoded samples and corresponding labels
    """
    dataset_path = _get_path("kddcup")
    data = pd.read_csv(dataset_path, header=0)

    # KDDCUP has very few "normal" samples.
    # As done in the state-of-the-art, the "normal" label is considered anomalous, everything else is considered normal.
    data["label"] = data["label"] == "normal"

    cat_cols = ["protocol_type", "service", "flag"]
    data[cat_cols] = OrdinalEncoder().fit_transform(data[cat_cols])

    y = data["label"].to_numpy(dtype=np.uint8)
    x = data.drop("label", axis=1).to_numpy(dtype=np.float32)

    return x, y


def load_dataset(dataset_name):
    """ Loads the required dataset.

    :param dataset_name: String (in _DATASET_FILES); dataset required
    :return: (np.ndarray(Float), np.ndarray(np.uint8));
        ordinal-encoded samples and corresponding labels (1 for anomalies, 0 for normal samples)
    """
    if dataset_name not in _DATASET_FILES:
        raise ValueError(f"Unkown dataset '{dataset_name}'. Available datasets are: {list(_DATASET_FILES)}.")

    if dataset_name == "kddcup":
        return _load_kddcup()

    dataset_path = _get_path(dataset_name)

    try:
        data = scipy.io.loadmat(dataset_path)
    except NotImplementedError:
        data = mat73.loadmat(dataset_path)

    x = data['X'].astype(np.float32)
    y = data['y'].astype(np.uint8).squeeze()

    return x, y


def normalize_data(fit_on, transform):
    """ Min-max-normalises samples from each array in "transform" based on "fit_on" samples.

    :param fit_on: np.ndarray(Float); samples on which to fit the normalisation
    :param transform: tuple(np.ndarray(Float)); arrays of samples to normalise
    :return: tuple(np.ndarray(Float));; arrays of normalised samples
    """
    normalizer = MinMaxScaler()
    normalizer.fit(fit_on)
    transformed = [normalizer.transform(x) for x in transform]
    return transformed


def generate_toy_data(contamination_rate, radius, seed=None):
    """ Generates a toy dataset. Normal samples are sampled from a standard gaussian distribution,
    anomalous samples are sampled from a noisy circle around the mean.
    There are 500 clean train samples, 1000 clean test samples and some anomalous test samples (see contamination_rate).

    :param contamination_rate: Float in [0, 1[; contamination rate of the test set
    :param radius: Float; radius of the anomalous circle
    :param seed: Int; seed for the dataset generation
    :return: (np.ndarray(Float), np.ndarray(Float), np.ndarray(np.uint8)); train set, test set, test labels
    """
    nb_train = 500
    nb_normal_test = 1000
    nb_anomaly_test = int(nb_normal_test * cont_to_anomalies_per_clean_sample(contamination_rate))

    rng = np.random.default_rng(seed=seed)
    x_train = rng.normal(size=(nb_train, 2))
    x_test = rng.normal(size=(nb_normal_test, 2))

    anomaly_angles = rng.uniform(0, 2 * np.pi, size=nb_anomaly_test)

    anomalies = radius * np.c_[np.cos(anomaly_angles), np.sin(anomaly_angles)]
    anomalies += rng.normal(scale=0.4, size=anomalies.shape)

    x_test = np.r_[x_test, anomalies]
    y_test = np.r_[np.zeros(nb_normal_test), np.ones(nb_anomaly_test)]

    return x_train.astype(np.float32), x_test.astype(np.float32), y_test.astype(np.uint8)

import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.io
import mat73

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

DATASET_URL = {
    "kddcup": "https://datahub.io/machine-learning/kddcup99/r/kddcup99.csv",
    "arrhythmia": "https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1",
    "thyroid": "https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1"
}

DATASET_FILE = {
    "kddcup": "kddcup99_csv.csv",
    "arrhythmia": "arrhythmia.mat",
    "thyroid": "thyroid.mat"
}

DATASET_FOLDER = os.path.abspath(os.path.join(__file__, "../../datasets/"))


def get_path(dataset_name):
    dataset_path = os.path.join(DATASET_FOLDER, DATASET_FILE[dataset_name])
    if not os.path.isfile(dataset_path):
        print(f"[DOWNLOAD] Download dataset {dataset_name} from {DATASET_URL[dataset_name]}")
        u = urllib.request.urlopen(DATASET_URL[dataset_name])
        with open(dataset_path, "wb") as f:
            f.write(u.read())
        u.close()

    return dataset_path


def load_kddcup():
    dataset_path = get_path("kddcup")
    data = pd.read_csv(dataset_path, header=0)

    # KDDCUP as very few "normal" sample
    # As done in the state-of-the-art, normal is considered as anomalous and everything else is normal
    data.loc[data["label"] != "normal", 'label'] = 0  # Normal samples
    data.loc[data["label"] == "normal", 'label'] = 1  # Anomalous samples

    data["protocol_type"] = OrdinalEncoder().fit_transform(data["protocol_type"].to_numpy().reshape(-1, 1))
    data["service"] = OrdinalEncoder().fit_transform(data["service"].to_numpy().reshape(-1, 1))
    data["flag"] = OrdinalEncoder().fit_transform(data["flag"].to_numpy().reshape(-1, 1))

    y = data["label"].to_numpy().astype(np.uint8)
    x = data.drop("label", axis=1).to_numpy().astype(np.float32)

    return x, y


def load_dataset(dataset_name):
    if dataset_name not in DATASET_FILE.keys():
        raise ValueError(f"Unkown dataset {dataset_name}. Available dataset are: {list(DATASET_FILE.keys())}")

    if dataset_name == "kddcup":
        return load_kddcup()

    dataset_path = get_path(dataset_name)

    try:
        data = scipy.io.loadmat(dataset_path)
    except NotImplementedError:
        data = mat73.loadmat(dataset_path)

    x = np.array(data['X']).astype(np.float32)
    y = np.array(data['y']).squeeze()

    return x, y


def normalize_data(fit_on, transform):
    normalizer = MinMaxScaler()
    normalizer.fit(fit_on)
    transformed = [normalizer.transform(x) for x in transform]
    return transformed

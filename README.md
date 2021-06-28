# Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol

This is the code of the paper *Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol*

## Environment setup

### Installation

We recommend you to use a virtual environment like *conda* or *virtualenv* and *virtualenv-wrapper*
The *requirements.txt* file contain all needed dependencies.
You can install them with:

`pip install -r requirements.txt`

### Datasets

Before running the experiments you need to download the datasets and put them in the folder *datasets*


## Reproducing the results



The script *f1-hack.py* compute the results shown in the Table 1 of the paper.
You can run the script by running:
`python f1-hack.py`

By default, KDD Cup is commented as it take a long time to run. 
You can change the dataset used by modifying the line 22 of *f1-hack*

You can also change the settings of the experiments by modifying the lines 24, 25, 26 (see details in paper)

## Reproduce the figures

Three notebooks are given to reproduce the figures of the paper.
You can launch a notebook with the command `jupyter notebook`

In each notebook you can change the *dataset_name* variable to choose the dataset on which to run the experiments.
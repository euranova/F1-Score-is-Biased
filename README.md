# Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol

This is the code of the paper *Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol*

## Environment setup

### Environment setup

The code is developped and tested with *Python 3.6.9* and *pip 21.1.3*.
We strongly recommand to use a virtual environment like *conda* or *virtualenv* (with *virtualenv-wrapper*). 

### Installation

The *requirements.txt* file contain all needed dependencies.
You can install all the packages by running the line:

`pip install -r requirements.txt`

### Datasets

By default datasets are not in the repository but are automatically downloaded the first time you run the code. Datasets are stored into the *datasets* folder.

### Jupyter Notebook

We provide different notebooks to reproduce our results.
You can lunch a notebook environment with the command:

`jupyter notebook`

## Reproducing the results

We provide multiple notebooks and a script to reproduce the results presented in the paper.

### Table 1 - Demonstration of the sensitivity of the metrics to the evaluation protocol

The script *f1-hack.py* produce the results presented in the Table 1.
You can run the script by running:
`python f1-hack.py`

By default, KDD Cup is commented as it take a long time to run. 
You can change the dataset used by modifying the line 22 of *f1-hack*

You can also change the settings of the experiments by modifying the lines 24, 25, 26 (see details in paper)

## Metrics per contamination estimation (Figure 1 & 2)

The notebook file *Metrics per contamination estimation (Fig2&3).ipynb* show the evolution of different metrics (F1-score, Precision, Recall) according to the estimated contamination rate.
It also produce an example of a ROC-Curve and a Precision Recall Curve.

You can change the dataset by changing the *dataset_name* variable in the second cell of the notebook.

## Metrics per true contamination rate (Figure 4)

The notebook file *Metrics per true contamination rate (Fig4).ipynb* show the evolution of the F1-score, AUC and AVPR according to the true contamination rate of the test set.

You can change the dataset by changing the *dataset_name* variable in the second cell of the notebook.


## Theoritical F1-Score (Figure 5 & 6)

The notebook file *Theoritical F1-Score (Fig5&6).ipynb* produce the figures 5 and 6 that present the theoricial evolution of the F1-score for varying contamination rate of the test set (Figure 5) and for different threshold (Figure 6)

## Toy Dataset Experiments (Figure 7)

The notebook file *Toy Dataset Experiments (Fig7).ipynb* reproduce the toy example illustration (Figure 7)


# Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol

This is the code used to produce the results of the paper
[Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol](https://arxiv.org/abs/2106.16020).

Please cite as 
```
@article{fourure2021anomaly,
  title={Anomaly Detection: How to Artificially Increase your F1-Score with a Biased Evaluation Protocol},
  author={Fourure, Damien and Javaid, Muhammad Usama and Posocco, Nicolas and Tihon, Simon},
  journal={arXiv preprint arXiv:2106.16020},
  year={2021}
}
```

## Environment Setup

### Python Environment

The code is developped and tested with *Python 3.6.9* and *pip 21.1.3*.
We strongly recommand to use a virtual environment like *conda* or *virtualenv* (with *virtualenv-wrapper*). 

### Installation

The *requirements.txt* file contain all needed dependencies.
You can install all the packages by running the following command line.

`pip install -r requirements.txt`

### Datasets

By default datasets are not in the repository but are automatically downloaded the first time you run the code. Datasets are saved in the *datasets* folder.

### Jupyter Notebook

We provide different notebooks to reproduce our results.
You can launch a notebook environment with the following command line.

`jupyter notebook`

## Reproducing the Results

We provide multiple notebooks and a script to reproduce the results presented in the paper.

### Table 1 - Demonstration of the Sensitivity of the Metrics to the Evaluation Protocol

![Table 1](./imgs/Table1.png)

The script *f1_hack.py* produces the results presented in Table 1.
It can be run using the following command line.

`python f1_hack.py`

By default, the results for the KDD Cup dataset are not computed as computing these can take up to several hours.
You can change the used datasets with the `datasets` argument.

`python f1_hack.py -d arrhythmia -d kddcup -d thyroid`

You can also change the settings of the experiments with the `settings` argument,
as explained when running
`python f1_hack.py --help`
(see details in the paper).

`python f1_hack.py -d arrhythmia -d kddcup -d thyroid -s 1.2e -s 2.2e -s 2.05e -s 2.05o`

## Metrics per Contamination Estimation (Figures 2 and 3)

![Figure 2](./imgs/Figure2.png)
![Figure 3](./imgs/Figure3.png)

The notebook file *Metrics per contamination estimation (Fig2&3).ipynb* shows the evolution of different metrics (F1-score, precision, recall) according to the estimated contamination rate.
It also produces an example of a ROC curve and a precision recall curve.

You can change the dataset by changing the *dataset_name* variable in the second cell of the notebook.

## Metrics per True Contamination Rate (Figure 4)

![Figure 4](./imgs/Figure4.png)

The notebook file *Metrics per true contamination rate (Fig4).ipynb* shows the evolution of F1-score, AUC and AVPR according to the true contamination rate of the test set.

You can change the dataset by changing the *dataset_name* variable in the second cell of the notebook.


## Theoretical F1-Score (Figures 5 and 6)

![Figure 5](./imgs/Figure5.png)
![Figure 6](./imgs/Figure6.png)

The notebook file *Theoretical F1-Score (Fig5&6).ipynb* produces Figures 5 and 6, which present the theoretical evolution of F1-score for different contamination rates of the test set (Figure 5) and for different thresholds (Figure 6).

## Toy Dataset Experiments (Figure 7)

![Figure 7](./imgs/Figure7.png)

The notebook file *Toy Dataset Experiments (Fig7).ipynb* reproduces the toy example illustrations (Figure 7).

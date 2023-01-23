# coding=utf8

import numpy as np
# zdefiniowanie klasyfikatorów, technik preprocessingu i metryk
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from strlearn.metrics import recall, precision, f1_score, geometric_mean_score_1
#walidacja krzyzowa
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

import matplotlib.pyplot as plt
from math import pi

import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA


clf = DecisionTreeClassifier(random_state=1410)
preprocs = {
    'none': None,
    'rus': RandomUnderSampler(random_state=1410),
    'cnn': CondensedNearestNeighbour(random_state=1410),
}
metrics = {
    "recall": recall,
    'precision': precision,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,

}


datasets = [ 'abalone19', 'cleveland-0_vs_4', 'ecoli-0-1-3-7_vs_2-6', 'ecoli1','ecoli4', 'glass-0-1-6_vs_5','glass2','glass4','glass5',
'pima', 'segment0', 'winequality-red-3_vs_5', 'winequality-red-4', 'winequality-red-8_vs_6-7','winequality-white-3_vs_7', 'winequality-white-3-9_vs_5',
 'winequality-white-9_vs_4', 'yeast-2_vs_4', 'yeast5', 'yeast6' ]

# datasets=['glass2']

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(preprocs), n_datasets, n_splits * n_repeats))


for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("./datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc in enumerate(preprocs):
            clf = clone(clf)

            if preprocs[preproc] == None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = preprocs[preproc].fit_resample(
                    X[train], y[train])

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])

            scores[preproc_id, data_id, fold_id] = precision(y[test], y_pred)

np.save('results_precision', scores)
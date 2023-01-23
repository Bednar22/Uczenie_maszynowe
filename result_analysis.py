# coding=utf8

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import ttest_rel
import scipy.stats as stats
from tabulate import tabulate
from scipy.stats import rankdata
from scipy.stats import ranksums


methods = {
    'None': "None",
    "RUS":"RUS",
    'CNN': "Cnn"
}

# Change file to analyse other metrics
scores = np.load('results_f1score.npy')
print("\nScores:\n", scores.shape)


# RANGI
mean_scores = np.mean(scores, axis=2).T
mean_scores_print = np.round(mean_scores, 3)
print("\nMean scores:\n", mean_scores_print)
# print("\nMean scores:\n", mean_scores)

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
print(methods)
print("\nMean ranks:\n", mean_ranks)


# PAROWE

alfa = .05
w_statistic = np.zeros((len(methods), len(methods)))
p_value = np.zeros((len(methods), len(methods)))

for i in range(len(methods)):
    for j in range(len(methods)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])


headers = list(methods.keys())
names_column = np.expand_dims(np.array(list(methods.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".3f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".3f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(methods), len(methods)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(methods), len(methods)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
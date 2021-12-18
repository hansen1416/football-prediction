import collections
import datetime
import logging
import os
import re
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans, DBSCAN
import warnings

from constants import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


metrics_table = {'EPL with ELO': {'RBF_SVM': {'Accuracy score': '0.61', 'Weighted F1 score': '0.55', 'Home win precision': 0.6312056737588653, 'Home win recall': 0.8018018018018018, 'Home win f1': 0.7063492063492063, 'Draw precision': 1.0, 'Draw recall': 0.022727272727272728, 'Draw f1': 0.044444444444444446, 'Away win precision': 0.5697674418604651, 'Away win recall': 0.6712328767123288, 'Away win f1': 0.6163522012578616}, 'Random_Forest': {'Accuracy score': '0.57', 'Weighted F1 score': '0.52', 'Home win precision': 0.6363636363636364, 'Home win recall': 0.7567567567567568, 'Home win f1': 0.691358024691358, 'Draw precision': 0.2, 'Draw recall': 0.022727272727272728, 'Draw f1': 0.04081632653061225, 'Away win precision': 0.5054945054945055, 'Away win recall': 0.6301369863013698, 'Away win f1': 0.5609756097560975}, 'LGBM': {'Accuracy score': '0.59', 'Weighted F1 score': '0.56', 'Home win precision': 0.6693548387096774, 'Home win recall': 0.7477477477477478, 'Home win f1': 0.7063829787234043, 'Draw precision': 0.3333333333333333, 'Draw recall': 0.11363636363636363, 'Draw f1': 0.16949152542372878, 'Away win precision': 0.5168539325842697, 'Away win recall': 0.6301369863013698, 'Away win f1': 0.5679012345679012}}, 'EPL without ELO': {'RBF_SVM': {'Accuracy score': '0.55', 'Weighted F1 score': '0.50', 'Home win precision': 0.583941605839416, 'Home win recall': 0.7207207207207207, 'Home win f1': 0.6451612903225806, 'Draw precision': 1.0, 'Draw recall': 0.022727272727272728, 'Draw f1': 0.044444444444444446, 'Away win precision': 0.4888888888888889, 'Away win recall': 0.6027397260273972, 'Away win f1': 0.5398773006134968}, 'Random_Forest': {'Accuracy score': '0.54', 'Weighted F1 score': '0.49', 'Home win precision': 0.5925925925925926, 'Home win recall': 0.7207207207207207, 'Home win f1': 0.6504065040650406, 'Draw precision': 0.2, 'Draw recall': 0.022727272727272728, 'Draw f1': 0.04081632653061225, 'Away win precision': 0.4772727272727273, 'Away win recall': 0.5753424657534246, 'Away win f1': 0.5217391304347826}, 'LGBM': {'Accuracy score': '0.57', 'Weighted F1 score': '0.55', 'Home win precision': 0.6201550387596899, 'Home win recall': 0.7207207207207207, 'Home win f1': 0.6666666666666666, 'Draw precision': 0.5384615384615384, 'Draw recall': 0.1590909090909091, 'Draw f1': 0.2456140350877193, 'Away win precision': 0.5, 'Away win recall': 0.589041095890411, 'Away win f1': 0.5408805031446541}}, 'ISA with ELO': {'RBF_SVM': {'Accuracy score': '0.54', 'Weighted F1 score': '0.48', 'Home win precision': 0.5106382978723404, 'Home win recall': 0.8372093023255814, 'Home win f1': 0.6343612334801763, 'Draw precision': 0.2, 'Draw recall': 0.034482758620689655, 'Draw f1': 0.0588235294117647, 'Away win precision': 0.6493506493506493, 'Away win recall': 0.5952380952380952, 'Away win f1': 0.6211180124223602}, 'Random_Forest': {'Accuracy score': '0.51', 'Weighted F1 score': '0.45', 'Home win precision': 0.4788732394366197, 'Home win recall': 0.7906976744186046, 'Home win f1': 0.5964912280701755, 'Draw precision': 0.3333333333333333, 'Draw recall': 0.034482758620689655, 'Draw f1': 0.0625, 'Away win precision': 0.5875, 'Away win recall': 0.5595238095238095, 'Away win f1': 0.573170731707317}, 'LGBM': {'Accuracy score': '0.51', 'Weighted F1 score': '0.49', 'Home win precision': 0.47107438016528924, 'Home win recall': 0.6627906976744186, 'Home win f1': 0.5507246376811594, 'Draw precision': 0.4074074074074074, 'Draw recall': 0.1896551724137931, 'Draw f1': 0.2588235294117647, 'Away win precision': 0.6, 'Away win recall': 0.5714285714285714, 'Away win f1': 0.5853658536585366}},
                 'ISA without ELO': {'RBF_SVM': {'Accuracy score': '0.47', 'Weighted F1 score': '0.40', 'Home win precision': 0.453416149068323, 'Home win recall': 0.8488372093023255, 'Home win f1': 0.5910931174089068, 'Draw precision': 0.0, 'Draw recall': 0.0, 'Draw f1': 0.0, 'Away win precision': 0.5573770491803278, 'Away win recall': 0.40476190476190477, 'Away win f1': 0.46896551724137925}, 'Random_Forest': {'Accuracy score': '0.49', 'Weighted F1 score': '0.44', 'Home win precision': 0.46206896551724136, 'Home win recall': 0.7790697674418605, 'Home win f1': 0.5800865800865801, 'Draw precision': 0.45454545454545453, 'Draw recall': 0.08620689655172414, 'Draw f1': 0.14492753623188406, 'Away win precision': 0.5416666666666666, 'Away win recall': 0.4642857142857143, 'Away win f1': 0.5}, 'LGBM': {'Accuracy score': '0.47', 'Weighted F1 score': '0.45', 'Home win precision': 0.45038167938931295, 'Home win recall': 0.686046511627907, 'Home win f1': 0.543778801843318, 'Draw precision': 0.35714285714285715, 'Draw recall': 0.1724137931034483, 'Draw f1': 0.23255813953488377, 'Away win precision': 0.5652173913043478, 'Away win recall': 0.4642857142857143, 'Away win f1': 0.5098039215686274}}, 'SLL with ELO': {'RBF_SVM': {'Accuracy score': '0.49', 'Weighted F1 score': '0.41', 'Home win precision': 0.5182926829268293, 'Home win recall': 0.8252427184466019, 'Home win f1': 0.6367041198501874, 'Draw precision': 0.375, 'Draw recall': 0.04285714285714286, 'Draw f1': 0.07692307692307691, 'Away win precision': 0.4107142857142857, 'Away win recall': 0.41818181818181815, 'Away win f1': 0.41441441441441446}, 'Random_Forest': {'Accuracy score': '0.46', 'Weighted F1 score': '0.42', 'Home win precision': 0.5347222222222222, 'Home win recall': 0.7475728155339806, 'Home win f1': 0.6234817813765182, 'Draw precision': 0.2, 'Draw recall': 0.08571428571428572, 'Draw f1': 0.12000000000000001, 'Away win precision': 0.4074074074074074, 'Away win recall': 0.4, 'Away win f1': 0.4036697247706423}, 'LGBM': {'Accuracy score': '0.48', 'Weighted F1 score': '0.45', 'Home win precision': 0.5503875968992248, 'Home win recall': 0.6893203883495146, 'Home win f1': 0.6120689655172414, 'Draw precision': 0.34375, 'Draw recall': 0.15714285714285714, 'Draw f1': 0.21568627450980393, 'Away win precision': 0.40298507462686567, 'Away win recall': 0.4909090909090909, 'Away win f1': 0.4426229508196721}}, 'SLL without ELO': {'RBF_SVM': {'Accuracy score': '0.46', 'Weighted F1 score': '0.37', 'Home win precision': 0.4915254237288136, 'Home win recall': 0.8446601941747572, 'Home win f1': 0.6214285714285713, 'Draw precision': 0.2, 'Draw recall': 0.014285714285714285, 'Draw f1': 0.026666666666666665, 'Away win precision': 0.3695652173913043, 'Away win recall': 0.3090909090909091, 'Away win f1': 0.33663366336633666}, 'Random_Forest': {'Accuracy score': '0.47', 'Weighted F1 score': '0.41', 'Home win precision': 0.525974025974026, 'Home win recall': 0.7864077669902912, 'Home win f1': 0.6303501945525293, 'Draw precision': 0.2727272727272727, 'Draw recall': 0.08571428571428572, 'Draw f1': 0.13043478260869562, 'Away win precision': 0.38461538461538464, 'Away win recall': 0.36363636363636365, 'Away win f1': 0.37383177570093457}, 'LGBM': {'Accuracy score': '0.48', 'Weighted F1 score': '0.45', 'Home win precision': 0.5390070921985816, 'Home win recall': 0.7378640776699029, 'Home win f1': 0.6229508196721312, 'Draw precision': 0.3870967741935484, 'Draw recall': 0.17142857142857143, 'Draw f1': 0.2376237623762376, 'Away win precision': 0.39285714285714285, 'Away win recall': 0.4, 'Away win f1': 0.39639639639639634}}}


def big_table():
    row_labels = []
    col_labels = []

    table_data = [[], [], [], [], [], []]
    i = 0

    for k1, v1 in metrics_table.items():
        row_labels.append(k1)
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                if i == 0:
                    col_labels.append(k2 + ' ' + k3)
                table_data[i].append("{:.2f}".format(float(v3)))
                # print(k3, v3)
        i += 1

    table_data = np.array(table_data)

    new_table_data = np.zeros((table_data.shape[1], table_data.shape[0]))
    new_table_data = new_table_data.tolist()

    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            new_table_data[j][i] = table_data[i][j]

    # print(row_labels)
    # print(col_labels)
    # print(table_data)
    # print(new_table_data)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(
        cellText=new_table_data,
        rowLabels=col_labels,
        colLabels=row_labels,
        # rowColours=["palegreen"] * 10,
        # colColours=["palegreen"] * 10,
        cellLoc='center',
        edges='horizontal',
        # visible_edges='horizontal',
        loc='upper left')

    # ax.set_title('matplotlib.axes.Axes.table() function Example',
    #              fontweight="bold")

    # plt.show()
    plt.tight_layout()
    plt.savefig("charts/classification_metrics.png", dpi=1000)


def league_comparison():
    # compare leagure data
    league_keys = {'EPL': ['EPL with ELO', 'EPL without ELO'],
                   'ISA': ['ISA with ELO', 'ISA without ELO'],
                   'SLL': ['SLL with ELO', 'SLL without ELO']}

    classifier = ['RBF_SVM', 'Random_Forest', 'LGBM']

    average_metrics = ['Accuracy score', 'Weighted F1 score']

    league_compare = {}

    for l, v1 in league_keys.items():
        league_compare[l] = {}
        for v2i in range(len(average_metrics)):
            score = 0
            for v1i in range(len(v1)):
                for v3i in range(len(classifier)):
                    # print(v1[v1i], classifier[v3i], average_metrics[v2i])
                    # print(metrics_table[v1[v1i]][classifier[v3i]][average_metrics[v2i]])
                    score += metrics_table[v1[v1i]
                                           ][classifier[v3i]][average_metrics[v2i]]

            league_compare[l][average_metrics[v2i]] = score / 6

    # print(league_compare)

    fig = plt.figure()
    X = np.arange(3)
    ax = fig.add_axes([0, 0, 1, 1])

    accuracy_score = [v['Accuracy score'] for k, v in league_compare.items()]
    f1_score = [v['Weighted F1 score'] for k, v in league_compare.items()]

    ax.bar(X + 0.00, accuracy_score, color='b',
           width=0.25, label="Accuracy score")
    ax.bar(X + 0.25, f1_score, color='g',
           width=0.25, label="Weighted F1 score")

    plt.xticks(X + 0.1, ['EPL', 'ISA', 'SLL'])
    # plt.title('Different league model performance')
    plt.legend()
    # plt.tight_layout()
    plt.savefig("charts/league_comparison.png", dpi=1000)


def league_elo_comparison():
    # compare leagure data
    league_elo_keys = ['EPL with ELO', 'EPL without ELO', 'ISA with ELO',
                       'ISA without ELO', 'SLL with ELO', 'SLL without ELO']

    classifier = ['RBF_SVM', 'Random_Forest', 'LGBM']

    average_metrics = ['Accuracy score', 'Weighted F1 score']

    league_elo_compare = {}

    for le in league_elo_keys:
        league_elo_compare[le] = {}
        for v2i in range(len(average_metrics)):
            score = []
            # for v1i in range(len(v1)):
            for v3i in range(len(classifier)):
                # print(v1[v1i], classifier[v3i], average_metrics[v2i])
                # print(metrics_table[v1[v1i]][classifier[v3i]][average_metrics[v2i]])
                score.append(
                    metrics_table[le][classifier[v3i]][average_metrics[v2i]])

            score = np.array(score)
            league_elo_compare[le][average_metrics[v2i]] = np.average(score)

    print(league_elo_compare)
    with_elo = ['EPL with ELO', 'ISA with ELO', 'SLL with ELO']
    without_elo = ['EPL without ELO', 'ISA without ELO', 'SLL without ELO']

    with_elo_acc = [league_elo_compare[x]['Accuracy score'] for x in with_elo]
    with_elo_f1 = [league_elo_compare[x]['Weighted F1 score']
                   for x in with_elo]
    without_elo_acc = [league_elo_compare[x]['Accuracy score']
                       for x in without_elo]
    without_elo_f1 = [league_elo_compare[x]['Weighted F1 score']
                      for x in without_elo]

    print((sum(with_elo_acc) - sum(without_elo_acc)) /
          3, (sum(with_elo_f1)-sum(without_elo_f1))/3)

    return

    fig = plt.figure()
    X = np.arange(6)
    ax = fig.add_axes([0, 0, 1, 1])

    accuracy_score = [v['Accuracy score']
                      for k, v in league_elo_compare.items()]
    f1_score = [v['Weighted F1 score'] for k, v in league_elo_compare.items()]

    ax.bar(X + 0.00, accuracy_score, color='b',
           width=0.25, label="Accuracy score")
    ax.bar(X + 0.25, f1_score, color='g',
           width=0.25, label="Weighted F1 score")

    plt.xticks(X + 0.1, ['EPL with ELO', 'EPL without ELO', 'ISA with ELO',
                         'ISA without ELO', 'SLL with ELO', 'SLL without ELO'])
    # plt.title('Different league model performance')
    plt.legend()
    # plt.tight_layout()
    plt.savefig("charts/league_elo_comparison.png", dpi=1000)


def elo_comparison():
    elo_keys = {'with': ['EPL with ELO', 'ISA with ELO', 'SLL with ELO'],
                'without': ['EPL without ELO', 'ISA without ELO', 'SLL without ELO']}

    classifier = ['RBF_SVM', 'Random_Forest', 'LGBM']

    metrics = ['Home win precision', 'Home win recall', 'Home win f1', 'Draw precision',
               'Draw recall', 'Draw f1', 'Away win precision', 'Away win recall', 'Away win f1']

    elo_compare = {}

    for w, v1 in elo_keys.items():
        elo_compare[w] = {}
        for v2i in range(len(metrics)):
            score = 0
            for v1i in range(len(v1)):
                for v3i in range(len(classifier)):
                    # print(v1[v1i], classifier[v3i], metrics[v2i])
                    score += metrics_table_num[v1[v1i]
                                               ][classifier[v3i]][metrics[v2i]]

            elo_compare[w][metrics[v2i]] = score / 9

    print(elo_compare)

    with_elo = list(elo_compare['with'].values())
    without_elo = list(elo_compare['without'].values())

    print(with_elo, without_elo)

    diff = [with_elo[i] - without_elo[i] for i in range(len(with_elo))]

    print(diff, sum(diff)/9)
    print(sum(diff[: 3])/3, sum(diff[3: 6])/3, sum(diff[6:])/3)

    fig = plt.figure()
    X = np.arange(len(with_elo))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.bar(X + 0.00, with_elo, color='b',
           width=0.25, label="With ELO")
    ax.bar(X + 0.25, without_elo, color='g',
           width=0.25, label="Without ELO")

    plt.xticks(X-0.1, ['Home win precision', 'Home win recall', 'Home win f1', 'Draw precision',
               'Draw recall', 'Draw f1', 'Away win precision', 'Away win recall', 'Away win f1'], rotation=45)
    # plt.title('Different league model performance')
    plt.legend()
    # plt.tight_layout()
    plt.savefig("charts/elo_comparison.png", dpi=1000)


if __name__ == "__main__":
    metrics_table_num = metrics_table.copy()

    for l1, v1 in metrics_table_num.items():
        for l2, v2 in v1.items():
            for l3, v3 in v2.items():
                metrics_table_num[l1][l2][l3] = float(v3)

    league_elo_comparison()

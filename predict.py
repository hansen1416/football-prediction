import collections
import datetime
import logging
import os
import re
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
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

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

CAT_COLS = ['league', 'match_link', 'Season', 'date',
            'score', 'label', 'spread', 'home', 'away', ]


def sort_dict(x, reverse=False):
    sorted_x = sorted(x.items(), key=lambda kv: kv[1], reverse=reverse)

    return collections.OrderedDict(sorted_x)


def read_match_metrics(weight_num=0, diff=True):

    data = None

    # for league in ['EPL', 'ISA', 'SLL']:
    for league in ['EPL']:
        for season in ['2018-2019', '2019-2020', '2020-2021']:

            metrics_file = season + league + \
                'weighted' + str(weight_num) + '.csv'

            df = pd.read_csv(os.path.join(
                DATASETS_DIR, 'weighted', metrics_file))

            if diff:
                num_cols = list(set(df.columns).difference(set(CAT_COLS)))

                home_cols = []
                away_cols = []

                for col in num_cols:
                    if col.startswith('home_'):
                        home_cols.append(col)
                    if col.startswith('away_'):
                        away_cols.append(col)

                        base_col = col[5:]

                        df[base_col] = df['home_' + base_col].copy() - \
                            df['away_' + base_col].copy()

                        # print(df[base_col], df['home_' + base_col],
                        #       df['away_' + base_col])
                        # exit()

                df = df.drop(columns=home_cols+away_cols)
                # print(df)
                # exit()

            elo_file = season + league + 'matches_elo.csv'
            elo = pd.read_csv(os.path.join(DATASETS_DIR, elo_file))
            elo = elo.drop(['date'], axis=1)
            df = df.merge(elo, on='match_link')

            if data is None:
                data = df.copy(deep=True)
            else:
                data = data.append(df.copy(deep=True))

    return data.reset_index(drop=True)


def select_n_features(x, y, n_features, features):

    select = SelectKBest(score_func=chi2, k=n_features)
    x_selected = select.fit_transform(x, y)

    filter = select.get_support()

    x_selected = pd.DataFrame(data=x_selected, columns=features[filter])

    feature_scores = dict(zip(features, select.scores_))

    feature_scores = sort_dict(feature_scores, reverse=True)

    return x_selected, features[filter], feature_scores


def pca_reduction(x, n_dimension):
    pca = PCA(n_components=n_dimension)
    return pca.fit_transform(x)


def split_scale_data(df):
    # print(df.dtypes)
    # remove xA xG data for now, we want basic player metrics
    # drop_cols = ['league', 'match_link', 'Season', 'date',
    #              'score', 'label', 'spread', 'home', 'away', ]
    # drop_cols = ['league', 'match_link', 'Season', 'date','score', 'label', 'spread',
    #  'home_Summery-Expected-xG', 'home_Summery-Expected-npxG', 'home_Summery-Expected-xA', 'home_Passing-xA',
    #  'away_Summery-Expected-xG', 'away_Summery-Expected-npxG', 'away_Summery-Expected-xA', 'away_Passing-xA']

    X = df.drop(CAT_COLS, axis=1)
    y = df['label']

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=46)

    # scale train data, must use min max to avoid negative data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X.columns)
    # use tain data scaler to transform test data
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=X.columns)

    X_train_feature_selected = []
    X_test_feature_selected = []

    # f_range = (80, 81)

    # for n in range(f_range[0], f_range[1]):
    #     x_selected, filetr, feature_scores = select_n_features(
    #         X_train_scaled, y_train, n, X.columns)

    #     X_train_feature_selected.append(x_selected)
    #     X_test_feature_selected.append(X_test_scaled[filetr])

    n_features = 80

    x_selected, filetr, feature_scores = select_n_features(
        X_train_scaled, y_train, n_features, X.columns)

    # count = 0
    # for k, v in feature_scores.items():
    #     print(k, v)
    #     count += 1
    #     if count > 20:
    #         break

    X_train_feature_selected.append(x_selected)
    X_test_feature_selected.append(X_test_scaled[filetr])

    # print('features selected {}-{}'.format(f_range[0], f_range[1]))
    print('features selected {}'.format(n_features))

    return X_train_feature_selected, X_test_feature_selected, y_train, y_test


def train(X_train_list, X_test_list, y_train, y_test, classifier, classifier_name):

    accuracies = []
    f1s = []

    for i in range(len(X_train_list)):

        classifier.fit(X_train_list[i], y_train)

        y_pred = classifier.predict(X_test_list[i])

        # print(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print("classifier {}, {} features, accuracy score {}, f1 score {}"
              .format(name, X_train_list[i].shape[1], accuracy, f1))

        accuracies.append(accuracy)
        f1s.append(f1)

    return accuracies, f1s


if __name__ == "__main__":

    weight_num = 0

    df = read_match_metrics(weight_num, diff=False)

    print("data shape {},{}, weight num {}".format(
        df.shape[0], df.shape[1], weight_num))

    X_train_feature_selected, X_test_feature_selected, y_train, y_test = split_scale_data(
        df)

    classifiers = {
        # "Neural_Net": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, alpha=1, max_iter=1000),
        # "RBF_SVM": SVC(kernel='rbf', C=1.5, gamma=0.15, random_state=46),
        "Random_Forest": RandomForestClassifier(criterion='entropy', random_state=46),
        # "LGBM": LGBMClassifier(max_depth=10, num_leaves=100, objective='multiclass', learning_rate=0.1,\
        #                        num_class=3, n_estimators=60, device_type='cpu', random_state=46),
    }

    accuracies = {}
    f1s = {}

    for name, clf in classifiers.items():

        accuracy_scores, f1_scores = train(X_train_feature_selected, X_test_feature_selected,
                                           y_train, y_test, clf, name)

        accuracies[name] = accuracy_scores
        f1s[name] = f1_scores

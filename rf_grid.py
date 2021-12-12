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
from predict import read_match_metrics, split_scale_data, classif_metrics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":

    leagues = [['EPL'], ['ISA'], ['SLL']]

    for league in leagues:
        for has_elo in [True]:

            weight_num = 6

            df = read_match_metrics(league, weight_num, with_elo=has_elo)

            print("data shape {},{}, weights {}".format(
                df.shape[0], df.shape[1], str(MATCH_WEIGHTS[weight_num])))

            X_train_feature_selected, X_test_feature_selected, y_train, y_test = split_scale_data(
                df, print_feature_scores=False)

            # Create the parameter grid based on the results of random search
            param_grid = {
                'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300, 1000]
            }
            # Create a based model
            rf = RandomForestClassifier()

            # Instantiate the grid search model
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                       cv=3, n_jobs=-1, verbose=2)

            grid_search.fit(X_train_feature_selected[0], y_train)

            print(grid_search.best_params_)

            best_grid = grid_search.best_estimator_

            y_pred = best_grid.predict(X_test_feature_selected[0])

            accuracy = "{:.2%}".format(accuracy_score(y_test, y_pred))
            f1_macro = "{:.2%}".format(
                f1_score(y_test, y_pred, average='macro'))
            f1_weighted = "{:.2%}".format(
                f1_score(y_test, y_pred, average='weighted'))

            print("Average accuracy score: {}, Macro f1 score: {}, Weighted f1 score: {}".format(
                accuracy, f1_macro, f1_weighted))

            metrics = classif_metrics(y_test, y_pred)

            print(metrics)

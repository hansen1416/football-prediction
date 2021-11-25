import json
import logging
import os
import string
import sys
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from constants import *

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

HISTORY_LENGTH = 5


def read_match_metrics():

    data = None

    for league in ['EPL']:
        for season in ['2018-2019', '2019-2020', '2020-2021']:

            filename = season + league + '-' + str(HISTORY_LENGTH) + '.csv'

            df = pd.read_csv(os.path.join(DATASETS_DIR, filename))

            if data is None:
                data = df
            else:
                data = data.append(df)

    return data.reset_index(drop=True)


def get_abs_corr_coef(X, y):
    """
    Compute

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Target variable

    Returns
    -------
    corr_coefs : numpy.ndarray of shape (n_features,)
        Vector of absolute values of correlation coefficients 
        for all features
    """
    # your code here
    corr_coefs = np.corrcoef(X, y, rowvar=False)

    return np.absolute(corr_coefs[:, -1][:-1])


if __name__ == "__main__":
    df = read_match_metrics()

    # print(df.dtypes)

    X = df.drop(['league', 'match_link', 'Season', 'date',
                'score', 'label', 'spread'], axis=1)
    y = df['label']

    n_features = 80

    scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler.fit_transform(X)

    select = SelectKBest(score_func=chi2, k=n_features)
    x_selected = select.fit_transform(X_scaled, y)

    print("After selecting best {} features: {}, reduced from origin {}".format(
        n_features, x_selected.shape, X.shape))

    filter = select.get_support()
    features = X.columns
    # print("All features:")
    # print(features)
    print("Selected best {}:".format(n_features))
    print(features[filter])
    # print(x_selected)

    names = [
        "Nearest Neighbors",
        # "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        # "LGBM",
        # "Gaussian Process",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        # LGBMClassifier(),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        x_selected, y, test_size=0.2, random_state=46)

    # print(X_train, y_test)

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)

        score = accuracy_score(y_test, y_pred)

        print("classifier {}, score {}".format(name, score))

    # feat_selector = SelectKBest(get_abs_corr_coef, k=40)

    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('feat_selector', feat_selector),
    #     ('model', RidgeClassifier()),
    # ])

    # search = GridSearchCV(pipeline,
    #                       #   {'model__alpha': np.arange(0.1, 10, 0.1)},
    #                       {'model__alpha': [3.2]},
    #                       cv=5, scoring="accuracy", verbose=3
    #                       )

    # search.fit(X_train, y_train)

    # print(search.best_params_)

    # coefficients = search.best_estimator_.named_steps['model'].coef_

    # print(coefficients.shape)

    # # importance = np.abs(coefficients)

    # # print(importance)

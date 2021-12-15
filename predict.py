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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

CAT_COLS = ['league', 'match_link', 'Season', 'date',
            'score', 'label', 'spread', ]


def sort_dict(x, reverse=False):
    sorted_x = sorted(x.items(), key=lambda kv: kv[1], reverse=reverse)

    return collections.OrderedDict(sorted_x)


def read_match_metrics(leagues, weight_num=0, diff=False, with_elo=True):
    """
    generate a dataframe for

    for how the data combined, 
    refer to https://github.com/hansen1416/football-prediction/blob/master/preprocess.py
    and https://github.com/hansen1416/football-prediction/blob/master/weighted.py
    """

    data = None

    for league in leagues:
        for season in ['2018-2019', '2019-2020', '2020-2021']:

            metrics_file = season + league + \
                'weighted' + str(weight_num) + '.csv'

            df = pd.read_csv(os.path.join(
                DATASETS_DIR, 'weighted', metrics_file))

            # calcuate the differece between home and away team, drop home/away specificed columns
            # this diff calculation turned out to be inefficient, think about why?
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

            # join elo with players metrics data
            # ELO collected from http://api.clubelo.com/'
            # using the rating before match start
            if with_elo:
                elo_file = season + league + 'matches_elo.csv'
                elo = pd.read_csv(os.path.join(DATASETS_DIR, elo_file))
                elo = elo.drop(['date', 'home', 'away'], axis=1)
                df = df.merge(elo, on='match_link')

            if data is None:
                data = df.copy(deep=True)
            else:
                data = data.append(df.copy(deep=True))

    return data.reset_index(drop=True)


def select_n_features(x, y, n_features, features):
    """
    select features by chi2
    """

    select = SelectKBest(score_func=chi2, k=n_features)
    # select = SelectKBest(score_func=f_regression, k=n_features)

    x_selected = select.fit_transform(x, y)

    filter = select.get_support()

    x_selected = pd.DataFrame(data=x_selected, columns=features[filter])

    feature_scores = dict(zip(features, select.scores_))

    feature_scores = sort_dict(feature_scores, reverse=True)

    return x_selected, features[filter], feature_scores


def split_scale_data(df, print_feature_scores=True):
    """
    split data to train and test,
    apply MinMaxScaler, SelectKBest, to train
    apply train scaler/mask to test
    """

    # drop_cols = ['home_Summery-Expected-xG', 'home_Summery-Expected-npxG', 'home_Summery-Expected-xA',
    #              'home_Passing-xA', 'away_Summery-Expected-xG', 'away_Summery-Expected-npxG',
    #              'away_Summery-Expected-xA', 'away_Passing-xA']

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

    # try different feature number
    # f_range = (30, 81)

    # for n in range(f_range[0], f_range[1]):
    #     x_selected, filetr, feature_scores = select_n_features(
    #         X_train_scaled, y_train, n, X.columns)

    #     X_train_feature_selected.append(x_selected)
    #     X_test_feature_selected.append(X_test_scaled[filetr])

    n_features = 80

    x_selected, filetr, feature_scores = select_n_features(
        X_train_scaled, y_train, n_features, X.columns)

    if print_feature_scores:
        count = 0
        for k, v in feature_scores.items():
            print(k, v)
            count += 1
            if count > 20:
                break
        # exit()

    X_train_feature_selected.append(x_selected)
    # apply the train mask to test data
    X_test_feature_selected.append(X_test_scaled[filetr])

    # print('features selected {}-{}'.format(f_range[0], f_range[1]))
    print('features selected {}'.format(n_features))

    return X_train_feature_selected, X_test_feature_selected, y_train, y_test


def train(X_train_list, X_test_list, y_train, y_test, classifier, classifier_name):
    """
    train and predict
    """

    home_win_mask = (y_test.values == 1)
    draw_mask = (y_test.values == 0)
    away_win_mask = (y_test.values == -1)

    for i in range(len(X_train_list)):

        classifier.fit(X_train_list[i], y_train)

        # print(classifier.feature_importances_)

        # y_train_pred = classifier.predict(X_train_list[i])

        # t_accuracy = "{:.2f}".format(accuracy_score(y_train, y_train_pred))
        # t_f1_macro = "{:.2f}".format(
        #     f1_score(y_train, y_train_pred, average='macro'))
        # t_f1_weighted = "{:.2f}".format(
        #     f1_score(y_train, y_train_pred, average='weighted'))

        # print("Classifier: {}, Average accuracy score: {}, Macro f1 score: {}, Weighted f1 score: {}. Training".format(
        #     classifier_name, t_accuracy, t_f1_macro, t_f1_weighted))

        y_pred = classifier.predict(X_test_list[i])

        # print(list(y_pred), list(y_test))

        accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred))
        # f1_macro = "{:.2f}".format(f1_score(y_test, y_pred, average='macro'))
        f1_weighted = "{:.2f}".format(
            f1_score(y_test, y_pred, average='weighted'))

        # print("Classifier: {}, Average accuracy score: {}, Macro f1 score: {}, Weighted f1 score: {}".format(
        #     classifier_name, accuracy, f1_macro, f1_weighted))

        result = {}
        result['Accuracy score'] = accuracy
        result['Weighted F1 score'] = f1_weighted

        scores = classif_metrics(y_test.values, y_pred)

        for l, score in scores.items():
            # print(l + ": " + str({k: "{:.2f}".format(v)
            #       for k, v in score.items()}))
            for k, v in score.items():
                result[l + ' ' + k] = v

        # print(result)

        return result

        y_pred_hw = y_pred[home_win_mask]
        y_pred_d = y_pred[draw_mask]
        y_pred_aw = y_pred[away_win_mask]

        print("Home win: {}/{}, Draw: {}/{}, Away win: {}/{}".format(
            len(y_pred_hw[y_pred_hw == 1]), len(y_pred_hw),
            len(y_pred_d[y_pred_d == 0]), len(y_pred_d), len(y_pred_aw[y_pred_aw == -1]), len(y_pred_aw)))

        print('-------------------------------')


def hwin_binary(np_array):
    return np.where(np_array == -1, 0, np_array)


def draw_binary(np_array):
    np_array = np.where(np_array == 0, 2, np_array)
    np_array = np.where(np_array != 2, 0, np_array)
    return np.where(np_array == 2, 1, np_array)


def awin_binary(np_array):
    np_array = np.where(np_array != -1, 0, np_array)
    return np.where(np_array == -1, 1, np_array)


def classif_metrics(y_true, y_pred):

    y_true_hw = hwin_binary(y_true)
    y_pred_hw = hwin_binary(y_pred)

    home_win = {'precision': precision_score(y_true_hw, y_pred_hw),
                'recall': recall_score(y_true_hw, y_pred_hw),
                'f1': f1_score(y_true_hw, y_pred_hw)}

    y_true_d = draw_binary(y_true)
    y_pred_d = draw_binary(y_pred)

    draw = {'precision': precision_score(y_true_d, y_pred_d),
            'recall': recall_score(y_true_d, y_pred_d),
            'f1': f1_score(y_true_d, y_pred_d)}

    y_true_aw = awin_binary(y_true)
    y_pred_aw = awin_binary(y_pred)

    away_win = {'precision': precision_score(y_true_aw, y_pred_aw),
                'recall': recall_score(y_true_aw, y_pred_aw),
                'f1': f1_score(y_true_aw, y_pred_aw)}

    return {'Home win': home_win, 'Draw': draw, 'Away win': away_win}


if __name__ == "__main__":

    leagues = [['EPL'], ['ISA'], ['SLL']]
    metrics_table = {}

    for league in leagues:
        for has_elo in [True, False]:

            if has_elo:
                table_title = league[0] + ' with ELO'
            else:
                table_title = league[0] + ' without ELO'

            metrics_table[table_title] = {}

            weight_num = 6

            df = read_match_metrics(league, weight_num, with_elo=has_elo)

            print("data shape {},{}, weights {}".format(
                df.shape[0], df.shape[1], str(MATCH_WEIGHTS[weight_num])))

            X_train_feature_selected, X_test_feature_selected, y_train, y_test = split_scale_data(
                df, print_feature_scores=False)

            classifiers = {
                "RBF_SVM": SVC(kernel='rbf', C=1.5, gamma=0.15, random_state=46),
                "Random_Forest": RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=31, random_state=46),
                "LGBM": LGBMClassifier(max_depth=3, min_data_in_leaf=12, num_leaves=20, objective='multiclass', learning_rate=0.1,
                                       num_class=3, n_estimators=60, device_type='cpu', random_state=46),
            }

            print("League: {}, With ELO: {}".format(str(league), str(has_elo)))

            for name, clf in classifiers.items():

                scores = train(X_train_feature_selected,
                               X_test_feature_selected, y_train, y_test, clf, name)

                metrics_table[table_title][name] = scores

            print("=====================================")

    print(metrics_table)

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
    # plt.savefig("classification_metrics.png", dpi=1000)

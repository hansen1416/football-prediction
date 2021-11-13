import os
import string
from functools import lru_cache

import numpy as np
import pandas as pd
import pickle

import constants

project_dir = os.path.abspath(os.path.join(__file__, os.pardir))

datadir = os.path.join(project_dir, 'datasets')

seasons = ['2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021']


def matches_data(seasons):
    matches_df = None

    for season in seasons:
        df = pd.read_csv(os.path.join(datadir, season + 'matches.csv'))
        df['Season'] = season

        if matches_df is None:
            matches_df = pd.DataFrame([], columns=df.columns)

        matches_df = matches_df.append(df)

    return matches_df


def match_players(seasons):
    dic = {}

    for season in seasons:
        d = np.load(os.path.join(datadir, season +
                    'match_players.npy'), allow_pickle='TRUE').item()

        dic.update(d)

    return dic


def players_data():

    # cache_file = "cache/playerdata.pkl"

    # if os.path.isfile(cache_file):

    #     with open(cache_file, 'rb') as f:
    #         print('read from cache')
    #         return pickle.load(f)

    cap = list(map(lambda x: x.upper(), string.ascii_uppercase))
    data = dict(zip(cap, [None] * len(cap)))

    for c in cap:
        data[c] = pd.read_csv(os.path.join(
            datadir, 'player_log_' + c + '.csv'))

    # with open(cache_file, "wb") as f:
    #     pickle.dump(data, f)

    return data


if __name__ == "__main__":

    r = players_data()

    print(r)

    # dic = match_players(seasons)

    # # dic.keys())
    # print(len(
    #     dic), dic['https://fbref.com/en/matches/18d42916/Burnley-Leeds-United-May-15-2021-Premier-League'])

    # player_data = pd.read_csv(
    #     constants.PLAYER_LOG_PREFIX + 'J.csv')

    # # print(player_data.head())

    # # print(player_data.shape)
    # # print(player_data.isna().sum().values)

    # # print(player_data['Season'].unique())

    # s1516 = player_data[player_data['Season'] == '2015-2016']
    # s1617 = player_data[player_data['Season'] == '2016-2017']
    # s1718 = player_data[player_data['Season'] == '2017-2018']
    # s1819 = player_data[player_data['Season'] == '2018-2019']
    # s1920 = player_data[player_data['Season'] == '2019-2020']
    # s2021 = player_data[player_data['Season'] == '2020-2021']

    # tmp = s2021

    # # print(dict(zip(tmp.columns, tmp.isna().sum().values)), tmp.shape)
    # print(tmp.isna().sum().values, tmp.shape)

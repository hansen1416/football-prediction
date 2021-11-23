import json
import logging
import os
import string
import sys
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import *

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

HISTORY_LENGTH = 5


def matches_data(seasons):
    matches_df = None

    for season in seasons:
        df = pd.read_csv(os.path.join(DATASETS_DIR, season + 'matches.csv'))
        df['Season'] = season

        if matches_df is None:
            matches_df = pd.DataFrame([], columns=df.columns)

        matches_df = matches_df.append(df)

    return matches_df


def match_players(seasons):
    dic = {}

    for season in seasons:
        d = np.load(os.path.join(DATASETS_DIR, season +
                    'match_players.npy'), allow_pickle='TRUE').item()

        dic.update(d)

    return dic


def players_data():

    cap = list(map(lambda x: x.upper(), string.ascii_uppercase))
    data = None

    for c in cap:
        df = pd.read_csv(os.path.join(
            DATASETS_DIR, 'player_log_' + c + '.csv'))

        if data is None:
            data = df
        else:
            data = data.append(df)

    data['Date'] = pd.to_datetime(data['Date'])

    return data.reset_index(drop=True)


def score_label(x):
    # print(x)
    spread = int(x[0]) - int(x[1])

    if spread > 0:
        return 1
    elif spread == 0:
        return 0
    else:
        return -1


def score_spread(x):
    return int(x[0]) - int(x[1])


def get_player_history(df, date, player_link, player_name):

    player_link = clean_url(player_link)

    # league = 'Premier League'

    # hist = clean_player_data(df)

    dp = list(map(int, date.split('-')))

    # hist = hist[(hist['Date'].dt.date < datetime.date(*dp)) \
    #             & (hist['PlayerUrl'] == player_link) \
    #             & (hist['Comp'] == league)]

    # hist = hist[(hist['Date'].dt.date < datetime.date(*dp)) \
    #             & (hist['PlayerUrl'] == player_link)]

    hist = df[(df['Date'].dt.date < datetime.date(*dp))
              & (df['PlayerUrl'] == player_link)]

    hist = hist.tail(HISTORY_LENGTH)
    hist = hist.drop(['Season', 'Date', 'Comp'], axis=1)
    hist = hist.reset_index(drop=True)

    logger.debug("Got player {}'s history, shape {}, {}".format(
        player_name, hist.shape[0], hist.shape[1]))

    return hist


def pad_players_data(df):
    if df.shape[0] >= HISTORY_LENGTH:
        return df

    empty_data = pd.DataFrame(index=range(
        HISTORY_LENGTH - df.shape[0]), columns=df.columns)

    return df.append(empty_data.fillna(0))


def combine_players_data(pdata_c, date, players_list):

    logger.info("combining {} players' data, {}".
                format(len(players_list), str([str(p[1]) + '-' + str(p[2]) + '-' + str(p[3]) for p in players_list])))

    nogk = None

    counter = 0

    for p in players_list:
        # only include players played more than 45 minutes
        if p[1] == '' or p[2] == '':
            logger.error('play time wrong, {}'.format(
                json.dumps({'info': p}, indent=4)))
            raise ValueError('play time wrong')

        if p[3] == '' or int(p[3]) < 45:
            continue

        # goal keeper's data is different
        if p[2] == 'GK':
            continue

        # print(pdata1821c.dtypes)

        df = get_player_history(pdata_c, date, p[0], p[1])

        df = pad_players_data(df)
        df = df.reset_index(drop=True)

        # print(df.columns[(df.dtypes != 'float64') & ~df.columns.isin(['PlayerUrl']) ].values)

        if nogk is None:
            nogk = df
        else:
            for col in nogk.columns:
                if col != 'PlayerUrl':
                    nogk[col] += df[col]

        counter += 1
        # only line ups
        if counter >= 10:
            break

    if nogk is None:
        logger.error("nogk is empty, {}"
                     .format(json.dumps({'date': date, 'player list': players_list}, indent=4)))
        raise ValueError('nogk is empty')

    nogk = nogk.drop(['PlayerUrl'], axis=1)

    return nogk.sum(axis=0)

    # frames = []

    # for i in range(nogk.shape[0]):
    #     df = nogk.iloc[i]
    #     df = df.add_suffix('_' + str(i))
    #     frames.append(df)

    # return pd.concat(frames, ignore_index=False)


def match_metrics(pdata_c, match_url, match_date):
    home_players = mpdata[match_url]['home_players']
    away_players = mpdata[match_url]['away_players']

    if len(home_players) < 11 or len(away_players) < 11:
        logger.error("match player data wrong, {}"
                     .format(json.dumps({match_url: mpdata[match_url]}, indent=4)))
        raise ValueError('match player data wrong')

    # todo distinguish keeper
    home_df = combine_players_data(pdata_c, match_date, home_players)

    away_df = combine_players_data(pdata_c, match_date, away_players)

    home_df = pd.DataFrame([home_df]).add_prefix('home_')
    away_df = pd.DataFrame([away_df]).add_prefix('away_')

    home_df['match_link'] = match_url
    away_df['match_link'] = match_url

    df = home_df.merge(away_df, on='match_link')

    logger.debug("got match data, url: {}, in shape {}, {}".format(
        match_url, df.shape[0], df.shape[1]))

    return df


def build_season_data(pdata_c, sdata, season):

    data = sdata[sdata['Season'] == season].copy(deep=True)

    metrics_df = None

    for _, row in data.iterrows():

        mdf = match_metrics(pdata_c, row['match_link'], row['date'])

        if metrics_df is None:
            metrics_df = mdf
        else:
            metrics_df = metrics_df.append(mdf)

        if metrics_df.isna().values.any():
            print(metrics_df)
            break

    return data.merge(metrics_df, on='match_link', how='left')


def test_player_history(mdata, mpdata, pdata, rand_n=100):

    testmdata = mdata.iloc[rand_n]
    match_link = testmdata['match_link']
    match_date = testmdata['date']

    sample_player = mpdata[match_link]['home_players'][0]

    print(match_link, match_date, sample_player)

    sample_player_hist = get_player_history(
        pdata, match_date, sample_player[0], sample_player[1])

    print(sample_player_hist.shape)
    print(sample_player_hist)


def test_match_metrics(mdata, pdata_c, rand_n=100):
    testmdata = mdata.iloc[rand_n]
    match_link = testmdata['match_link']
    match_date = testmdata['date']

    print(match_metrics(pdata_c, match_link, match_date))


if __name__ == "__main__":

    seasons = ['2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021']

    mdata = matches_data(seasons)
    mpdata = match_players(seasons)
    pdata = players_data()

    # check match links, they should be the same
    assert sorted(mdata['match_link'].values) == sorted(list(
        mpdata.keys())), "match links not match in match data and match players data"

    sdata = mdata[['match_link', 'Season', 'date']].copy(deep=True)

    sdata['score'] = mdata['score'].str.split('–')  # .copy(deep=True)

    sdata = sdata.reset_index(drop=True)

    sdata['label'] = sdata['score'].apply(score_label)
    sdata['spread'] = sdata['score'].apply(score_spread)

    for season in seasons:
        season_sdata = sdata[sdata['Season'] == season]
        label_dis = season_sdata['label'].value_counts()  # 1,0,-1
        print(season, label_dis[1] / label_dis.sum(),
              label_dis[-1] / label_dis.sum())

    """
    Home advantage is about 45%-49% in seasons before the covid-19 pandemic, away win is between 28%-34%.
    In season 2020-2021 during the pandemic, home win percentage dropped to below 38%, 
    while away win rise to 40%. The impact of lack of attendance.
    """

    cat_cols = ['Min', 'Result', 'Venue', 'PlayerName',
                'Opponent', 'Pos', 'Start', 'Squad', 'Day', 'Round']

    pdata1821 = pdata[pdata['Season'].isin(
        ['2018-2019', '2019-2020', '2020-2021'])]

    # count nan in numeric data

    drop_cols = []

    # nan_threshold = 30000
    # for index, value in pdata1821.isnull().sum().sort_values().items():
    #     if value > nan_threshold:
    #         drop_cols.append(index)

    keep_cols = list(set(pdata.columns).difference(set(cat_cols + drop_cols)))

    pdata1821 = pdata1821[~pdata1821['Date'].isna()]

    pdata1821c = pdata1821[keep_cols].reset_index(drop=True).fillna(0)

    print(pdata1821.shape, pdata1821c.shape)

    # test_player_history(mdata, mpdata, pdata)

    # test_match_metrics(mdata, pdata1821c)

    for season in ['2018-2019', '2019-2020', '2020-2021']:
        data_season = build_season_data(pdata1821c, sdata, season)

        filename = season + '-' + str(HISTORY_LENGTH) + '-PL.csv'

        data_season.to_csv(os.path.join(DATASETS_DIR, filename), index=False)

        print(pd.read_csv(os.path.join(DATASETS_DIR, filename)))

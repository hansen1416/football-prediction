import os

import pandas as pd

import constants

project_dir = os.path.abspath(os.path.join(__file__, os.pardir))

datadir = os.path.join(project_dir, 'datasets')

player_data = pd.read_csv(
    constants.PLAYER_LOG_PREFIX + 'J.csv')

# print(player_data.head())


# print(player_data.shape)
# print(player_data.isna().sum().values)

# print(player_data['Season'].unique())

s1516 = player_data[player_data['Season'] == '2015-2016']
s1617 = player_data[player_data['Season'] == '2016-2017']
s1718 = player_data[player_data['Season'] == '2017-2018']
s1819 = player_data[player_data['Season'] == '2018-2019']
s1920 = player_data[player_data['Season'] == '2019-2020']
s2021 = player_data[player_data['Season'] == '2020-2021']

tmp = s2021

# print(dict(zip(tmp.columns, tmp.isna().sum().values)), tmp.shape)
print(tmp.isna().sum().values, tmp.shape)

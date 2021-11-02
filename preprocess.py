import os

import pandas as pd

import constants

project_dir = os.path.abspath(os.path.join(__file__, os.pardir))

datadir = os.path.join(project_dir, 'datasets')

player_data = pd.read_csv(
    constants.PLAYER_LOG_PREFIX + 'J.csv')

print(player_data.head())

print(player_data.columns)

import csv
import os
import sys
import logging

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.indexes import base
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, WebDriverException

from constants import *
from data_columns import *

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def fetch_players(match_players_file):

    match_players = np.load(match_players_file, allow_pickle='TRUE').item()

    for _, players in match_players.items():

        for p in players['home_players']:
            feach_match_logs(p)
            break
        # for p in players['away_players']:
        #     feach_match_logs(p)

        break


def feach_match_logs(players_info):

    player_url, player_name = players_info

    # print(player_url, player_name)

    log_file_name = PLAYER_LOG_PREFIX + player_name[0] + '.csv'

    all_columns = columns_basic + columns_summary + columns_passing + \
        columns_passing_types + columns_gca + \
        columns_defense + columns_possession + columns_misc

    if os.path.isfile(log_file_name):
        match_logs = pd.read_csv(log_file_name)
    else:
        match_logs = pd.DataFrame([], columns=all_columns)

    # we fetched the data already
    # if match_logs.get(player_url):
    #     return

    seasons = ['2016-2017', '2017-2018', '2018-2019', '2019-2020']
    data_type = ['summary', 'passing', 'passing_types',
                 'gca', 'defense', 'possession', 'misc']

    season_urls = 'https://fbref.com/en/players/3201b03d/matchlogs/2016-2017/summary/Danny-Simpson-Match-Logs'

    fire_fox_service = Service(FIREFOX_DRIVER_PATH)

    driver = webdriver.Firefox(service=fire_fox_service)

    driver.get(season_urls)

    matchlogs_all = driver.find_element(By.ID, 'matchlogs_all')

    only_summary = True

    summary_data: pd.DataFrame = None

    for row in matchlogs_all.find_elements(By.CSS_SELECTOR, 'tbody tr:not(.spacer):not(.thead):not(.hidden)'):

        row_data = [td.text for td in row.find_elements(
            By.CSS_SELECTOR, 'th,td')]

        if len(row_data) == 29:

            data = pd.DataFrame([[player_url, player_name] + row_data[:28]],
                                columns=columns_basic + columns_summary_short)

            if summary_data is None:
                summary_data = data
            else:
                summary_data = summary_data.append(data, ignore_index=True)

        else:

            only_summary = False
            # print(row_data)
            data = pd.DataFrame([[player_url, player_name] + row_data[:37]],
                                columns=columns_basic + columns_summary)

            if summary_data is None:
                summary_data = data
            else:
                summary_data = summary_data.append(data, ignore_index=True)

    # print(summary_data)
    # exit()

    driver.quit()

    if only_summary:

        match_logs = match_logs.append(summary_data, ignore_index=True)

        match_logs.to_csv(log_file_name, index=False)
    else:
        pass
    # todo, fetach detailed data


match_players_file = 'datasets/1617match_players.npy'

fetch_players(match_players_file)

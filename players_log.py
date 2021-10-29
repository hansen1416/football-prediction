# import csv
import os
import re
import sys
import logging
from queue import Queue

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
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

    q = Queue()

    match_players = np.load(match_players_file, allow_pickle='TRUE').item()

    for _, players in match_players.items():

        for p in players['home_players']:
            q.put(p)
        for p in players['away_players']:
            q.put(p)

    return q


match_players_file = 'datasets/1718match_players.npy'

player_queue = fetch_players(match_players_file)

while player_queue.qsize() > 0:
    print(player_queue.get())


def fetach_summary_data(player_url, player_name, season):

    url = player_url + '/matchlogs/{}/summary/'.format(season)

    logger.info('start fetching summary data for {} from {} '.format(
        player_name, player_url))

    driver = webdriver.Firefox(service=Service(FIREFOX_DRIVER_PATH))

    driver.get(url)

    matchlogs_all = driver.find_element(By.ID, 'matchlogs_all')

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

            # print(row_data)
            data = pd.DataFrame([[player_url, player_name] + row_data[:37]],
                                columns=columns_basic + columns_summary)

            if summary_data is None:
                summary_data = data
            else:
                summary_data = summary_data.append(data, ignore_index=True)

    driver.quit()

    return summary_data


def fetach_advanced_data(player_url, player_name, season, block_name):

    block_dict = {'passing': (11, 32, columns_passing), 'passing_types': (11, 36, columns_passing_types),
                  'gca': (11, 25, columns_gca), 'defense': (11, 34, columns_defense),
                  'possession': (11, 35, columns_possession), 'misc': (11, 27, columns_misc)}

    url = player_url + '/matchlogs/{}/{}/'.format(season, block_name)

    logger.info('start fetching {} data for {} from {}'.format(
        block_name, player_name, player_url))

    driver = webdriver.Firefox(service=Service(FIREFOX_DRIVER_PATH))

    driver.get(url)

    matchlogs_all = driver.find_element(By.ID, 'matchlogs_all')

    result: pd.DataFrame = None

    for row in matchlogs_all.find_elements(By.CSS_SELECTOR, 'tbody tr:not(.spacer):not(.thead):not(.hidden)'):

        row_data = [td.text for td in row.find_elements(
            By.CSS_SELECTOR, 'th,td')]

        data = pd.DataFrame([row_data[block_dict[block_name][0]:block_dict[block_name][1]]],
                            columns=block_dict[block_name][2])

        if result is None:
            result = data
        else:
            result = result.append(data, ignore_index=True)

    driver.quit()

    return result


def feach_match_logs(player_info):

    player_url, player_name, position, _ = player_info

    # print(player_url, player_name)
    # exit()

    player_url = re.sub(r'/[^/]+$', '', player_url)
    # goal keeper
    if position == 'GK':
        return

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

    summary_data = fetach_summary_data(player_url, player_name, seasons[1])

    if len(summary_data.columns) == 30:

        match_logs = match_logs.append(summary_data, ignore_index=True)

        match_logs.to_csv(log_file_name, index=False)

    else:
        passing_data = fetach_advanced_data(
            player_url, player_name, seasons[2], 'passing')

        passing_types_data = fetach_advanced_data(
            player_url, player_name, seasons[2], 'passing_types')

        print(passing_types_data)

        gca_data = fetach_advanced_data(
            player_url, player_name, seasons[2], 'gca')

        defense_data = fetach_advanced_data(
            player_url, player_name, seasons[2], 'defense')

        possession_data = fetach_advanced_data(
            player_url, player_name, seasons[2], 'possession')

        misc_data = fetach_advanced_data(
            player_url, player_name, seasons[2], 'misc')

        all_data = pd.concat([summary_data, passing_data, passing_types_data, gca_data,
                              defense_data, possession_data, misc_data], axis=1)

        # print(all_data)
        match_logs = match_logs.append(all_data, ignore_index=True)

        match_logs.to_csv(log_file_name, index=False)

    # todo, fetach detailed data


# fetach_passing_data('https://fbref.com/en/players/336dbcb2',
#                     'Adama Diomande', '2017-2018')

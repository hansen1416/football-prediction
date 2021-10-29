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

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def build_players_queue(match_players_file):

    q = Queue()

    match_players = np.load(match_players_file, allow_pickle='TRUE').item()

    for _, players in match_players.items():

        for p in players['home_players']:
            q.put(p)
        for p in players['away_players']:
            q.put(p)

    return q


def fetach_summary_data(player_url, player_name, season):

    url = player_url + '/matchlogs/{}/summary/'.format(season)

    logger.info('start fetching summary data for {} from {}, in season {}'.format(
        player_name, player_url, season))

    try:
        driver = webdriver.Firefox(service=Service(FIREFOX_DRIVER_PATH))

        driver.get(url)

        matchlogs_all = driver.find_element(By.ID, 'matchlogs_all')

        summary_data: pd.DataFrame = None

        for row in matchlogs_all.find_elements(By.CSS_SELECTOR, 'tbody tr:not(.spacer):not(.thead):not(.hidden)'):

            row_data = [td.text for td in row.find_elements(
                By.CSS_SELECTOR, 'th,td')]

            if len(row_data) == 29:

                data = pd.DataFrame([[player_url, player_name, season] + row_data[:28]],
                                    columns=columns_basic + columns_summary_short)

                if summary_data is None:
                    summary_data = data
                else:
                    summary_data = summary_data.append(data, ignore_index=True)

            else:

                # print(row_data)
                data = pd.DataFrame([[player_url, player_name, season] + row_data[:37]],
                                    columns=columns_basic + columns_summary)

                if summary_data is None:
                    summary_data = data
                else:
                    summary_data = summary_data.append(data, ignore_index=True)
    finally:
        driver.quit()

    return summary_data


def fetach_advanced_data(player_url, player_name, season, block_name):

    block_dict = {'passing': (11, 32, columns_passing), 'passing_types': (11, 36, columns_passing_types),
                  'gca': (11, 25, columns_gca), 'defense': (11, 34, columns_defense),
                  'possession': (11, 35, columns_possession), 'misc': (11, 27, columns_misc)}

    url = player_url + '/matchlogs/{}/{}/'.format(season, block_name)

    logger.info('start fetching {} data for {} from {}, in season {}'.format(
        block_name, player_name, player_url, season))

    try:
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
    finally:
        driver.quit()

    return result


if __name__ == "__main__":

    match_players_file = os.path.join('datasets', '1718match_players.npy')

    player_queue = build_players_queue(match_players_file)

    counter = 0

    while player_queue.qsize() > 0:

        url, name, pos, min = player_queue.queue[0]

        url = re.sub(r'/[^/]+$', '', url)

        # we deal goal keeper differently
        if pos == 'GK':
            player_queue.get()
            continue

        log_file_name = PLAYER_LOG_PREFIX + strip_accents(name[0]) + '.csv'

        if not os.path.isfile(log_file_name):
            all_columns = columns_basic + columns_summary + columns_passing + \
                columns_passing_types + columns_gca + \
                columns_defense + columns_possession + columns_misc

            empty_data = pd.DataFrame([], columns=all_columns)

            with open(log_file_name, 'w') as f:
                empty_data.to_csv(f, index=False)

        df = pd.read_csv(log_file_name)

        season_queue = Queue()
        [season_queue.put(t) for t in ['2015-2016', '2016-2017',
                                       '2017-2018', '2018-2019', '2019-2020', '2020-2021']]

        while season_queue.qsize() > 0:

            season = season_queue.queue[0]

            season_df = df.loc[(df['PlayerUrl'] == url) &
                               (df['Season'] == season)]

            # we already players info in this season
            if season_df.shape[0] > 0:
                season_queue.get()
                continue

            try:
                summary_data = fetach_summary_data(url, name, season)

                if len(summary_data.columns) == 31:

                    summary_data = pd.DataFrame([], columns=all_columns).append(
                        summary_data, ignore_index=True)

                    with open(log_file_name, 'a') as f:
                        summary_data.to_csv(f, header=False)

                    logger.info("add player {} {}, season {} summary data to {}".format(
                        url, name, season, log_file_name))

                else:

                    advanced_data = []

                    for block in ['passing', 'passing_types', 'gca', 'defense', 'possession', 'misc']:
                        advanced_data.append(
                            fetach_advanced_data(url, name, season, block))

                    full_data = pd.concat(
                        [summary_data] + advanced_data, axis=1)

                    with open(log_file_name, 'a') as f:
                        full_data.to_csv(f, header=False)

                    logger.info("add player {} {}, season {} full data to {}".format(
                        url, name, season, log_file_name))

            except WebDriverException:
                # in case of crawling failed, we don't remove item from queue
                continue
            except NoSuchElementException:
                # it means there is not data for this player in this season, just pass
                logger.info(
                    "no data for player {} in season".format(url, season))
                pass

            season_queue.get()

        player_queue.get()

        logger.info(
            "player {} {}, data is saved for all season".format(url, name))

        counter += 1

        if counter > 1:
            break

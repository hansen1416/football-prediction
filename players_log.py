# import csv
from constants import *
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By
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

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

opts = webdriver.FirefoxOptions()
# opts.add_argument("--width=1024")
# opts.add_argument("--height=678")
opts.add_argument("--headless")


def build_players_queue(match_players_files):

    q = Queue()

    for file in match_players_files:

        match_players = np.load(file, allow_pickle='TRUE').item()

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
        driver = webdriver.Firefox(service=Service(
            FIREFOX_DRIVER_PATH), options=opts)

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

        if summary_data is None or summary_data.shape[0] == 0:
            raise NoSuchElementException

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
        driver = webdriver.Firefox(service=Service(
            FIREFOX_DRIVER_PATH), options=opts)

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


def fetch_keeper_data(player_url, player_name, season):

    url = player_url + '/matchlogs/{}/keeper/'.format(season)

    logger.info('start fetching keeper data for {} from {}, in season {}'.format(
        player_name, player_url, season))

    try:
        driver = webdriver.Firefox(service=Service(
            FIREFOX_DRIVER_PATH), options=opts)

        driver.get(url)

        matchlogs_all = driver.find_element(By.ID, 'matchlogs_all')

        result: pd.DataFrame = pd.DataFrame(
            [], columns=columns_basic + columns_keeper)

        for row in matchlogs_all.find_elements(By.CSS_SELECTOR, 'tbody tr:not(.spacer):not(.thead):not(.hidden)'):

            row_data = [td.text for td in row.find_elements(
                By.CSS_SELECTOR, 'th,td')]

            if len(row_data) == 21:
                data = pd.DataFrame([[player_url, player_name, season] + row_data[:20]],
                                    columns=columns_basic + columns_keeper_short)
            else:
                data = pd.DataFrame([[player_url, player_name, season] + row_data[:36]],
                                    columns=columns_basic + columns_keeper)

            result = result.append(data, ignore_index=True)

        if result is None or result.shape[0] == 0:
            raise NoSuchElementException

    finally:
        driver.quit()

    return result


if __name__ == "__main__":

    all_columns = columns_basic + columns_summary + columns_passing + \
        columns_passing_types + columns_gca + \
        columns_defense + columns_possession + columns_misc

    match_players_files = [os.path.join('datasets', '1617match_players.npy'),
                           os.path.join('datasets', '1718match_players.npy'),
                           os.path.join('datasets', '1819match_players.npy'),
                           os.path.join('datasets', '1920match_players.npy'),
                           os.path.join('datasets', '2021match_players.npy')]

    player_queue = build_players_queue(match_players_files)

    logger.info('build players queue of size {}'.format(player_queue.qsize()))

    no_data_season_file = os.path.join('logs', 'no_data_season.csv')

    if not os.path.isfile(no_data_season_file):

        with open(no_data_season_file, 'w') as f:
            pd.DataFrame([], columns=['PlayerUrl', 'Season']
                         ).to_csv(f, index=False)
    # for some there is no data in some season, save it so we don't have to check it again
    no_data_df = pd.read_csv(no_data_season_file)

    counter = 0

    keeper_log_file = os.path.join('datasets', 'keeper_log.csv')

    keeper_df = pd.read_csv(keeper_log_file)

    while player_queue.qsize() > 0:

        url, name, pos, min = player_queue.queue[0]

        url = re.sub(r'/[^/]+$', '', url)

        log_file_name = PLAYER_LOG_PREFIX + strip_accents(name[0]) + '.csv'

        if not os.path.isfile(log_file_name):

            empty_data = pd.DataFrame([], columns=all_columns)

            with open(log_file_name, 'w') as f:
                empty_data.to_csv(f, index=False)

        df = pd.read_csv(log_file_name)

        season_queue = Queue()
        [season_queue.put(t) for t in ['2015-2016', '2016-2017',
                                       '2017-2018', '2018-2019', '2019-2020', '2020-2021']]

        while season_queue.qsize() > 0:

            season = season_queue.queue[0]

            no_df = no_data_df[(no_data_df['PlayerUrl'] == url) &
                               (no_data_df['Season'] == season)]

            if no_df.shape[0] > 0:
                season_queue.get()
                continue

            # we deal goal keeper differently
            if pos == 'GK':
                season_df = keeper_df.loc[(keeper_df['PlayerUrl'] == url) &
                                          (keeper_df['Season'] == season)]

                # we already players info in this season
                if season_df.shape[0] > 0:
                    season_queue.get()
                    continue

                try:
                    keeper_data = fetch_keeper_data(url, name, season)

                    with open(keeper_log_file, 'a') as f:
                        keeper_data.to_csv(f, index=False, header=False)

                    logger.info("add keeper {} season {} data to {}".format(
                        url, season, keeper_log_file))

                except NoSuchElementException:
                    # it means there is not data for this player in this season, just pass
                    with open(no_data_season_file, 'a') as f:
                        pd.DataFrame([[url, season]], columns=[
                            'PlayerUrl', 'Season']).to_csv(f, header=False, index=False)

                    logger.warning(
                        "no keeper data for player {} in season {}".format(url, season))
                except WebDriverException:
                    # in case of crawling failed, we just skip it, try fetch it later
                    logger.error(
                        'fetchinf keeper data from {} in season {} failed'.format(url, season))
                finally:
                    season_queue.get()
            # non-keeper players
            else:
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

                except NoSuchElementException:
                    # it means there is not data for this player in this season, just pass
                    with open(no_data_season_file, 'a') as f:
                        pd.DataFrame([[url, season]], columns=[
                            'PlayerUrl', 'Season']).to_csv(f, header=False, index=False)

                    logger.warning(
                        "no data for player {} in season {}".format(url, season))
                except WebDriverException:
                    # in case of crawling failed, we just skip it, try fetch it later
                    logger.error(
                        'fetchinf data from {} in season {} failed'.format(url, season))
                finally:
                    season_queue.get()

        player_queue.get()

        logger.info(
            "player {} {}, data is saved for all season".format(url, name))

        counter += 1

        if counter > 240:
            break

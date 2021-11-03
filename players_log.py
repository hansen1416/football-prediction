# import csv
from threading import Thread
from pandas.core.indexes.base import Index
from constants import *
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By
import os
import re
import string
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


def clean_url(url):
    return re.sub(r'/[^/]+$', '',  url)


class Worker(Thread):

    def __init__(self, queue, thread_name):
        Thread.__init__(self)
        self.queue: Queue = queue
        self.thread_name = thread_name

    def run(self):
        counter = 0

        while self.queue.qsize() > 0:

            url, name, pos = player_queue.get()

            try:

                log_file_name = PLAYER_LOG_PREFIX + \
                    strip_accents(name[0]) + '.csv'
            except IndexError:
                logger.error(
                    "IndexError url is {}, name is {}".format(url, name))
                continue

            season_queue = Queue()
            [season_queue.put(t) for t in ['2015-2016', '2016-2017',
                                           '2017-2018', '2018-2019', '2019-2020', '2020-2021']]

            while season_queue.qsize() > 0:

                season = season_queue.get()

                # we deal goal keeper differently
                if pos == 'GK':

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
                        continue
                # non-keeper players
                else:

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
                        continue

            logger.info(
                "{}/{}, player {} {}, data is saved for all season".format(counter, total, url, name))

            counter += 1

            if counter > 100:
                break


if __name__ == "__main__":

    all_columns = columns_basic + columns_summary + columns_passing + \
        columns_passing_types + columns_gca + \
        columns_defense + columns_possession + columns_misc

    names = np.array(list(string.ascii_uppercase))

    existing_players = set()

    for C in names:
        fn = PLAYER_LOG_PREFIX + C + '.csv'

        if not os.path.isfile(fn):

            empty_data = pd.DataFrame([], columns=all_columns)

            with open(fn, 'w') as f:
                empty_data.to_csv(f, index=False)

        ep_df = pd.read_csv(fn)

        ep_df = ep_df[['PlayerUrl', 'Season']]

        ep_tuples = [tuple(x) for x in ep_df.to_numpy()]

        existing_players.update(ep_tuples)

    # fetched keeper data
    keeper_log_file = os.path.join('datasets', 'keeper_log.csv')

    if not os.path.isfile(keeper_log_file):

        empty_data = pd.DataFrame([], columns=all_columns)

        with open(keeper_log_file, 'w') as f:
            empty_data.to_csv(f, index=False)

    keeper_df = pd.read_csv(keeper_log_file)

    kp_tuples = [tuple(x)
                 for x in keeper_df[['PlayerUrl', 'Season']].to_numpy()]

    existing_players.update(kp_tuples)

    # there is no data for some season
    no_data_season_file = os.path.join('logs', 'no_data_season.csv')

    if not os.path.isfile(no_data_season_file):

        with open(no_data_season_file, 'w') as f:
            pd.DataFrame([], columns=['PlayerUrl', 'Season']
                         ).to_csv(f, index=False)

    # for some there is no data in some season, save it so we don't have to check it again
    no_data_df = pd.read_csv(no_data_season_file)

    nd_tuples = [tuple(x)
                 for x in no_data_df[['PlayerUrl', 'Season']].to_numpy()]

    existing_players.update(nd_tuples)

    seasons = ['2015-2016', '2016-2017', '2017-2018',
               '2018-2019', '2019-2020', '2020-2021']

    # match_players_files = [os.path.join(
    #     'datasets', s + 'match_players.npy') for s in seasons[1:]]

    player_queue = Queue()

    for s in seasons[1:]:

        file = os.path.join('datasets', s + 'match_players.npy')

        match_players = np.load(file, allow_pickle='TRUE').item()

        for _, players in match_players.items():
            try:
                for p in players['home_players']:
                    url = clean_url(p[0])

                    item = (url, p[1], p[2])

                    if (url, s) not in existing_players and item not in list(player_queue.queue):
                        player_queue.put(item)

                for p in players['away_players']:
                    url = clean_url(p[0])

                    item = (url, p[1], p[2])

                    if (url, s) not in existing_players and item not in list(player_queue.queue):
                        player_queue.put(item)

            except IndexError:
                logger.info('name error, {} {}'.format(p[0], p[1]))

    total = player_queue.qsize()

    logger.info('build players queue of size {}'.format(total))

    # print(player_queue.queue)

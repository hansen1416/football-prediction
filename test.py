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
opts.add_argument("--headless")


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

        if result is None:
            raise NoSuchElementException

    finally:
        driver.quit()

    return result


# res = fetch_keeper_data('https://fbref.com/en/players/53af52f3/',
#                         'Kasper Schmeichel', '2015-2016')


# print(res)

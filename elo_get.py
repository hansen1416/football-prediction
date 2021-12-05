import csv
import os
import re
import sys
import logging
import requests
from codecs import iterdecode

import numpy as np
import pandas as pd
from selenium.webdriver.firefox.webdriver import WebDriver
# from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
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

opts = webdriver.FirefoxOptions()
opts.add_argument("--headless")

if __name__ == "__main__":

    leagues = ['EPL', 'SLL', 'ISA']
    seasons = ['2018-2019', '2019-2020', '2020-2021']

    for l in leagues:
        for s in seasons:
            match_history_csv = os.path.join(
                DATASETS_DIR, s + l + 'matches.csv')

            with open(match_history_csv, mode='r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    home_team = ''.join(row['home'].split())
                    away_team = ''.join(row['away'].split())

                    for team in [home_team, away_team]:

                        if team in elomapping:
                            team = elomapping[team]

                        elofile = os.path.join(
                            DATASETS_DIR, 'elo', team + '.csv')

                        if os.path.isfile(elofile):
                            continue

                        with requests.get(CLUBELO_URL + team, stream=True) as elocsv:

                            decoded_content = elocsv.content.decode('utf-8')

                            with open(elofile, 'w') as ef:
                                ef.write(decoded_content)

                                logger.info(
                                    "download elo for {}, to {}".format(team, elofile))

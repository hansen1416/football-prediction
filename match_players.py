import csv
import os
import re
import sys
import logging

import numpy as np
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


def fetch_players(match_url, home_id, away_id):
    # fireFoxService = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    fire_fox_service = Service(FIREFOX_DRIVER_PATH)

    try:
        driver = webdriver.Firefox(service=fire_fox_service)

        driver.get(match_url)

        home_table = driver.find_element(
            By.ID, 'stats_' + home_id + '_summary')

        away_table = driver.find_element(
            By.ID, 'stats_' + away_id + '_summary')

        # print(home_table, away_table)

        home_players = []

        for row in home_table.find_elements(By.CSS_SELECTOR, 'tbody tr'):
            tds = [td for td in row.find_elements(By.CSS_SELECTOR, 'th, td')]

            player_a = tds[0].find_element(By.CSS_SELECTOR, 'a')

            home_players.append((player_a.get_attribute('href'),
                                player_a.text, tds[3].text, tds[5].text))

        away_players = []

        for row in away_table.find_elements(By.CSS_SELECTOR, 'tbody tr'):
            tds = [td for td in row.find_elements(By.CSS_SELECTOR, 'th, td')]

            player_a = tds[0].find_element(By.CSS_SELECTOR, 'a')

            away_players.append((player_a.get_attribute('href'),
                                player_a.text, tds[3].text, tds[5].text))

        # print(home_players)
        # print(away_players)
    finally:
        driver.quit()

    return {'home_players': home_players, 'away_players': away_players}


def match_players(match_history_csv, save_filename):

    if os.path.isfile(save_filename):
        match_players = np.load(save_filename, allow_pickle='TRUE').item()
    else:
        match_players = {}

    # print(match_players)
    # return

    counter = 0

    with open(match_history_csv, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if not match_players.get(row['match_link']):
                try:

                    home_re_group = re.match(
                        "https://fbref.com/en/squads/([\d\w]+)/", row['home_link'])

                    away_re_group = re.match(
                        "https://fbref.com/en/squads/([\d\w]+)/", row['away_link'])

                    home_id = home_re_group.group(1)

                    away_id = away_re_group.group(1)

                    match_players[row['match_link']] = fetch_players(
                        row['match_link'], home_id, away_id)
                except WebDriverException:
                    logging.error(row['match_link'] + ' fetch failed')
                    continue

            counter += 1
            logging.info('fetch data row %d' % counter)
            if counter >= 30:
                break

    # print(counter)
    # print(match_players)

    np.save(save_filename, match_players)

    return match_players


for d in [('datasets/1617matches.csv', 'datasets/1617match_players.npy'),
          ('datasets/1718matches.csv', 'datasets/1718match_players.npy'),
          ('datasets/1819matches.csv', 'datasets/1819match_players.npy'),
          ('datasets/1920matches.csv', 'datasets/1920match_players.npy'),
          ('datasets/2021matches.csv', 'datasets/2021match_players.npy'), ]:
    match_players(d[0], d[1])

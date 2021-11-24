import csv
import os
import re
import sys
import logging

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


def fetch_players(match_url, home_id, away_id):

    try:
        driver = browser_driver()

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
            # some link might fail, fix it
            # if row['match_link'] == 'https://fbref.com/en/matches/d27f6889/Stoke-City-Tottenham-Hotspur-September-10-2016-Premier-League':

            #     home_re_group = re.match(
            #         "https://fbref.com/en/squads/([\d\w]+)/", row['home_link'])

            #     away_re_group = re.match(
            #         "https://fbref.com/en/squads/([\d\w]+)/", row['away_link'])

            #     home_id = home_re_group.group(1)

            #     away_id = away_re_group.group(1)

            #     match_players[row['match_link']] = fetch_players(
            #         row['match_link'], home_id, away_id)
            #     # exit()

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
            # if counter >= 8000:
            #     break

    # print(counter)
    # print(match_players)

    np.save(save_filename, match_players)

    return match_players


def get_team_id(link):
    re_group = re.match(
        "https://fbref.com/en/squads/([\d\w]+)/", link)

    return re_group.group(1)


def check_missing(leagues, seasons=['2018-2019', '2019-2020', '2020-2021']):

    for l in leagues:
        for s in seasons:

            matches = pd.read_csv(os.path.join(
                DATASETS_DIR, s + l + 'matches.csv'))

            filename = os.path.join(DATASETS_DIR, s + l + 'match_players.npy')
            match_players = np.load(filename, allow_pickle='TRUE').item()

            missing_links = set()

            for match_link, item in match_players.items():
                for p in item['home_players']:
                    if p[1] == '' or p[2] == '':
                        missing_links.add(match_link)

                for p in item['away_players']:
                    if p[1] == '' or p[2] == '':
                        missing_links.add(match_link)

            for match_link in list(missing_links):

                match_row = matches[matches['match_link'] == match_link]

                # print(match_row['home_link'].values[0],
                #       match_row['away_link'].values[0])

                home_id = get_team_id(match_row['home_link'].values[0])
                away_id = get_team_id(match_row['away_link'].values[0])

                logging.info('fetch missing data for %s' % match_link)
                # print(match_link, p, home_id, away_id)
                data = fetch_players(match_link, home_id, away_id)

                match_players[match_link] = data

            np.save(filename, match_players)

            logging.info('save match players data to %s' % filename)


if __name__ == "__main__":

    leagues = ['EPL', 'SLL', 'ISA']
    seasons = ['2018-2019', '2019-2020', '2020-2021']

    for l in leagues:
        for s in seasons:
            m = os.path.join(DATASETS_DIR, s + l + 'matches.csv')
            mp = os.path.join(DATASETS_DIR, s + l + 'match_players.npy')
            match_players(m, mp)

    check_missing(leagues, seasons)

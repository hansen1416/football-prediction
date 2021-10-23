import csv
import os
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

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def fetch_players(match_url):
    # fireFoxService = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    fire_fox_service = Service(
        '/home/hlz/.wdm/drivers/geckodriver/linux64/v0.30.0/geckodriver')

    driver = webdriver.Firefox(service=fire_fox_service)

    driver.get(match_url)

    home = driver.find_element(By.CSS_SELECTOR, '#a.lineup')

    away = driver.find_element(By.CSS_SELECTOR, '#b.lineup')
    # fetch the data for all 18 players, but we only use starting 11 players later
    home_players = [(a.get_attribute('href'), a.text)
                    for a in home.find_elements(By.CSS_SELECTOR, 'td a')]

    away_players = [(a.get_attribute('href'), a.text)
                    for a in away.find_elements(By.CSS_SELECTOR, 'td a')]

    # print(home_players)
    # print(away_players)

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
                    match_players[row['match_link']] = fetch_players(
                        row['match_link'])
                except WebDriverException:
                    logging.error(row['match_link'] + ' fetch failed')
                    continue

            counter += 1
            logging.info('fetch data row %d' % counter)
            # if counter >= 1:
            #     break

    # print(counter)
    # print(match_players)

    np.save(save_filename, match_players)

    return match_players


match_history_csv = 'datasets/1617matches.csv'
save_filename = 'datasets/1617match_players.npy'

match_history_csv = 'datasets/1718matches.csv'
save_filename = 'datasets/1718match_players.npy'

mps = match_players(match_history_csv, save_filename)

# print(mps)
# print(mps['https://fbref.com/en/matches/78c3fc92/Hull-City-Leicester-City-August-13-2016-Premier-League'])

import csv
import os
import sys
import logging

import numpy as np
# from selenium.webdriver.firefox.webdriver import WebDriver
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.firefox import GeckoDriverManager
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


def fetch_players(match_players_file):

    match_players = np.load(match_players_file, allow_pickle='TRUE').item()

    for _, players in match_players.items():

        for p in players['home_players']:
            feach_match_logs(p)
        for p in players['away_players']:
            feach_match_logs(p)

        break


def feach_match_logs(players_info):

    player_url, player_name = players_info

    # print(player_url, player_name)

    log_file_name = 'datasets/match_log_{}.npy'.format(player_name[0])

    if os.path.isfile(log_file_name):
        match_logs = np.load(log_file_name, allow_pickle='TRUE').item()
    else:
        match_logs = {}

    # we fetched the data already
    if match_logs.get(player_url):
        return

    seasons = ['2016-2017', '2017-2018', '2018-2019', '2019-2020']
    data_type = ['summary', 'passing', 'passing_types',
                 'gca', 'defense', 'possession', 'misc']

    season_urls = [
        'https://fbref.com/en/players/3201b03d/matchlogs/2016-2017/passing/Danny-Simpson-Match-Logs']

    fire_fox_service = Service(
        '/home/hlz/.wdm/drivers/geckodriver/linux64/v0.30.0/geckodriver')

    driver = webdriver.Firefox(service=fire_fox_service)

    driver.get(season_urls[0])

    matchlogs_all = driver.find_element(By.ID, 'matchlogs_all')

    thead = matchlogs_all.find_elements(
        By.CSS_SELECTOR, 'tr')[1]

    # summary_keys = ['Date', 'Day', 'Comp', 'Round', 'Venue', 'Result', 'Squad', 'Opponent', 'Start', 'Pos', 'Min', 'Gls', 'Ast',
    #                 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'Int', 'OG', 'PKwon', 'PKcon', 'Match Report']

    passing_keys = ['Cmp', 'Att', 'Cmp%', 'TotDist', 'PrgDist', 'Cmp', 'Att', 'Cmp%', 'Cmp', 'Att',
                    'Cmp%', 'Cmp', 'Att', 'Cmp%', 'Ast', 'xA', 'KP', '1/3', 'PPA', 'CrsPA', 'Prog', 'Match Report']

    # misc_keys = ['CrdY', 'CrdR',
    #             '2CrdY', 'Fls', 'Fld', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'Recov', 'Won', 'Lost', 'Won%', 'Match Report']

    summary_keys = [td.text for td in thead.find_elements(
        By.CSS_SELECTOR, 'th')]

    print(summary_keys)

    # away = driver.find_element(By.CSS_SELECTOR, '#b.lineup')
    # # fetch the data for all 18 players, but we only use starting 11 players later
    # home_players = [(a.get_attribute('href'), a.text)
    #                 for a in home.find_elements(By.CSS_SELECTOR, 'td a')]

    # away_players = [(a.get_attribute('href'), a.text)
    #                 for a in away.find_elements(By.CSS_SELECTOR, 'td a')]

    # # print(home_players)
    # # print(away_players)

    # driver.quit()

    # return {'home_players': home_players, 'away_players': away_players}


match_players_file = 'datasets/1617match_players.npy'

fetch_players(match_players_file)

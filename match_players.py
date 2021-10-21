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
from selenium.common.exceptions import NoSuchElementException

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

matches = []

match_history_csv = 'datasets/1617matches.csv'


def fetch_players(match_url):
    # fireFoxService = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    fire_fox_service = Service(
        '/home/hlz/.wdm/drivers/geckodriver/linux64/v0.30.0/geckodriver')

    driver = webdriver.Firefox(service=fire_fox_service)

    driver.get(match_url)

    home = driver.find_element(By.CSS_SELECTOR, '#a.lineup')

    away = driver.find_element(By.CSS_SELECTOR, '#b.lineup')

    home_players = [{a.get_attribute('href'): a.text}
                    for a in home.find_elements(By.CSS_SELECTOR, 'td a')]

    away_players = [{a.get_attribute('href'): a.text}
                    for a in away.find_elements(By.CSS_SELECTOR, 'td a')]

    print(home_players)
    print(away_players)

    driver.quit()


def match_players(match_history_csv):

    with open(match_history_csv, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            fetch_players(row['match_link'])
            break


match_players(match_history_csv)

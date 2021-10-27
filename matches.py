import csv
import os
import sys
import logging
from selenium.webdriver.firefox.webdriver import WebDriver
# from pprint import pprint

# from selenium.webdriver import Chrome
# from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from constants import *

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def featch_season_history(url, element_id, filename):

    # fireFoxService = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    fire_fox_service = Service(FIREFOX_DRIVER_PATH)

    driver = webdriver.Firefox(service=fire_fox_service)

    driver.get(url)

    table: WebDriver = driver.find_element(By.ID, element_id)
    match_history = []

    # iterate over all the rows
    for row in table.find_elements(By.CSS_SELECTOR, "tbody tr:not(.spacer)"):

        try:
            info = {}
            match_time = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="time"]')
            info['time'] = match_time.text
        except NoSuchElementException:
            continue

        try:
            keys = ['week', 'day', 'date', 'time', 'home', 'home_xg', 'score', 'away_xg',
                    'away', 'attendance', 'venue', 'referee', 'match_report']

            info.update(
                dict(zip(keys, [td.text for td in row.find_elements(By.CSS_SELECTOR, 'th,td')])))

            home_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="squad_a"] a')
            info['home_link'] = home_link.get_attribute('href')

            away_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="squad_b"] a')
            info['away_link'] = away_link.get_attribute('href')

            match_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="score"] a')
            info['match_link'] = match_link.get_attribute('href')

            report_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="match_report"] a')
            info['report_link'] = report_link.get_attribute('href')
        except NoSuchElementException:
            pass
        except Exception as e:
            logger.error('Iterating table exception: ' + str(e))

        match_history.append(info)

    # print(match_history)

    driver.quit()

    try:
        with open('datasets/' + filename, 'w') as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=list(match_history[0].keys()))
            writer.writeheader()
            for data in match_history:
                writer.writerow(data)
    except IOError:
        print("I/O error")


url = 'https://fbref.com/en/comps/9/1526/schedule/2016-2017-Premier-League-Scores-and-Fixtures'
element_id = "sched_1526_1"
filename = "1617matches.csv"

# url = 'https://fbref.com/en/comps/9/1631/schedule/2017-2018-Premier-League-Scores-and-Fixtures'
# element_id = "sched_1631_1"
# filename = "1718matches.csv"

# url = "https://fbref.com/en/comps/9/1889/schedule/2018-2019-Premier-League-Scores-and-Fixtures"
# element_id = "sched_1889_1"
# filename = "1819matches.csv"

featch_season_history(url, element_id, filename)

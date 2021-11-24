import csv
import os
import sys
import logging

import pandas as pd
from pandas.core.indexes.base import Index
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from constants import *

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def featch_season_history(league, url, element_id, filename):

    # if element_id == 'sched_1526_1':
    #     columns = ['week', 'day', 'date', 'time', 'home', 'score', 'away', 'attendance', 'venue',
    #                'referee', 'match_report', 'home_link', 'away_link', 'match_link', 'report_link']
    # else:
    columns = ['league', 'week', 'day', 'date', 'time', 'home', 'home_xg', 'score', 'away_xg', 'away', 'attendance',
               'venue', 'referee', 'match_report', 'home_link', 'away_link', 'match_link', 'report_link']

    driver = browser_driver()

    driver.get(url)

    log_file = os.path.join(DATASETS_DIR, filename)

    empty_data = pd.DataFrame([], columns=columns)
    # by doing this, we emptyed the file
    with open(log_file, 'w') as f:
        empty_data.to_csv(f, index=False)

    table: WebDriver = driver.find_element(By.ID, element_id)
    match_history = []

    counter = 1

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

            data = [league] + [td.text for td in row.find_elements(
                By.CSS_SELECTOR, 'th,td')]

            # we don't need the last "note" column
            data.pop()

            home_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="squad_a"] a')
            # info['home_link'] = home_link.get_attribute('href')
            data.append(home_link.get_attribute('href'))

            away_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="squad_b"] a')
            # info['away_link'] = away_link.get_attribute('href')
            data.append(away_link.get_attribute('href'))

            match_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="score"] a')
            # info['match_link'] = match_link.get_attribute('href')
            data.append(match_link.get_attribute('href'))

            report_link = row.find_element(
                By.CSS_SELECTOR, 'td[data-stat="match_report"] a')
            # info['report_link'] = report_link.get_attribute('href')
            data.append(report_link.get_attribute('href'))

            df = pd.DataFrame([data], columns=columns)

            with open(log_file, 'a') as f:
                df.to_csv(f, header=False, index=False)

            counter += 1
            if counter % 10 == 0:
                logger.info("saving data row {}".format(counter))

        except NoSuchElementException:
            pass
        except Exception as e:
            logger.error('Iterating table exception: ' + str(e))

        match_history.append(info)

    driver.quit()


if __name__ == "__main__":

    leagues = [

    ]

    for l in leagues:

        filename = l['season'] + l['league'] + 'matches.csv'

        featch_season_history(l['league'], l['url'], l['element_id'], filename)


# url = 'https://fbref.com/en/comps/9/1526/schedule/2016-2017-Premier-League-Scores-and-Fixtures'
# element_id = "sched_1526_1"
# filename = "2016-2017matches.csv"

# url = 'https://fbref.com/en/comps/9/1631/schedule/2017-2018-Premier-League-Scores-and-Fixtures'
# element_id = "sched_1631_1"
# filename = "2017-2018matches.csv"

# url = "https://fbref.com/en/comps/9/1889/schedule/2018-2019-Premier-League-Scores-and-Fixtures"
# element_id = "sched_1889_1"
# filename = "2018-2019matches.csv"

# url = "https://fbref.com/en/comps/9/3232/schedule/2018-2019-Premier-League-Scores-and-Fixtures"
# element_id = "sched_3232_1"
# filename = "2019-2020matches.csv"

# url = "https://fbref.com/en/comps/9/10728/schedule/2020-2021-Premier-League-Scores-and-Fixtures"
# element_id = "sched_10728_1"
# filename = "2020-2021matches.csv"

# {
#     'url': 'https://fbref.com/en/comps/12/1886/schedule/2018-2019-La-Liga-Scores-and-Fixtures',
#     'element_id': "sched_1886_1",
#     'season': '2018-2019',
#     'league': 'SLL',
# },
# {
#     'url': 'https://fbref.com/en/comps/12/3239/schedule/2019-2020-La-Liga-Scores-and-Fixtures',
#     'element_id': "sched_3239_1",
#     'season': '2019-2020',
#     'league': 'SLL',
# },
# {
#     'url': 'https://fbref.com/en/comps/12/10731/schedule/2020-2021-La-Liga-Scores-and-Fixtures',
#     'element_id': "sched_10731_1",
#     'season': '2020-2021',
#     'league': 'SLL',
# },
# {
#     'url': 'https://fbref.com/en/comps/11/1896/schedule/2018-2019-Serie-A-Scores-and-Fixtures',
#     'element_id': "sched_1896_1",
#     'season': '2018-2019',
#     'league': 'ISA',
# },
# {
#     'url': 'https://fbref.com/en/comps/11/3260/schedule/2019-2020-Serie-A-Scores-and-Fixtures',
#     'element_id': "sched_3260_1",
#     'season': '2019-2020',
#     'league': 'ISA',
# },
# {
#     'url': 'https://fbref.com/en/comps/11/10730/schedule/2020-2021-Serie-A-Scores-and-Fixtures',
#     'element_id': "sched_10730_1",
#     'season': '2020-2021',
#     'league': 'ISA',
# },

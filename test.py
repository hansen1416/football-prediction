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


if __name__ == "__main__":
    match_players_files = [os.path.join('datasets', '1617match_players.npy'),
                           os.path.join('datasets', '1718match_players.npy'),
                           os.path.join('datasets', '1819match_players.npy'),
                           os.path.join('datasets', '1920match_players.npy'),
                           os.path.join('datasets', '2021match_players.npy')]

    q = set()

    for file in match_players_files:

        match_players = np.load(file, allow_pickle='TRUE').item()

        for _, players in match_players.items():

            for p in players['home_players']:
                q.add(p[0])
            for p in players['away_players']:
                q.add(p[0])

    print(len(q), q)

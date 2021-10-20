import os
import sys
import logging
# from pprint import pprint

# from selenium.webdriver import Chrome
# from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By

# log to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


# todo, first collect all players id and name
playerHash = {'id': '21a66f6a', 'name': 'Harry-Kane'}


# fireFoxService = webdriver.Firefox(executable_path=GeckoDriverManager().install())
fireFoxService=Service('/home/hlz/.wdm/drivers/geckodriver/linux64/v0.30.0/geckodriver')

driver = webdriver.Firefox(service=fireFoxService)

driver.get("https://fbref.com/en/players/21a66f6a/matchlogs/2017-2018/summary/Harry-Kane-Match-Logs")

# https://fbref.com/en/players/5eae500a/matchlogs/2017-2018/summary/Romelu-Lukaku-Match-Logs

# print(driver.current_url)

table = driver.find_element(By.ID, "matchlogs_all")

print(table)

# driver.quit()
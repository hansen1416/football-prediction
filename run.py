import logging
from pprint import pprint



#Or use the context manager
from selenium import webdriver
# from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By

# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

# driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
s=Service('/home/hlz/.wdm/drivers/geckodriver/linux64/v0.30.0/geckodriver')

with webdriver.Firefox(service=s) as driver:
# #your code inside this indent
    print('with chrome driver')
    driver.get("https://fbref.com/en/players/21a66f6a/matchlogs/2017-2018/summary/Harry-Kane-Match-Logs")

    # print(driver.current_url)

    table = driver.find_element(By.ID, "matchlogs_all")

    print(table)
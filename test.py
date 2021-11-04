# import csv
import time
import random
from threading import Thread
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


class Worker(Thread):

    def __init__(self, queue, thread_name):
        Thread.__init__(self)
        self.queue: Queue = queue
        self.thread_name = thread_name

    def run(self):
        while self.queue.qsize() > 0:
            # Get the work from the queue and expand the tuple
            i = self.queue.get()
            # try:
            pause = random.randint(1, 3)
            print(self.thread_name, i, 'sleep {}'.format(pause))
            time.sleep(pause)
            # finally:
            #     self.queue.task_done()


if __name__ == "__main__":
    que = Queue()

    for i in range(100):
        que.put(i)

    for n in range(10):
        worker = Worker(que, 'thread-' + str(n))
        # worker.daemon = True
        worker.start()
from players_log import fetach_summary_data
from selenium.common.exceptions import NoSuchElementException

try:
    fetach_summary_data('https://fbref.com/en/players/77e39b04/',
                        'Francisco Trincao', '2017-2018')
# except NoSuchElementException:
#     print('no data')
finally:
    print(123123)

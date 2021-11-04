import os
import unicodedata

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

if os.name == 'nt':
    BROWSER_DRIVER_PATH = r'C:\Users\hanse\.wdm\drivers\chromedriver\win32\95.0.4638.54\chromedriver.exe'
else:
    # FIREFOX_DRIVER_PATH = '/home/hlz/.wdm/drivers/geckodriver/linux64/v0.30.0/geckodriver'
    BROWSER_DRIVER_PATH = '/home/hlz/soccer-data/driver/geckodriver'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PLAYER_LOG_PREFIX = os.path.join(PROJECT_DIR, 'datasets', 'player_log_')

columns_basic = ['PlayerUrl', 'PlayerName', 'Season', 'Date', 'Day', 'Comp', 'Round', 'Venue',
                 'Result', 'Squad', 'Opponent', 'Start', 'Pos', 'Min']

columns_summary = ['Summery-Performance-Gls', 'Summery-Performance-Ast', 'Summery-Performance-PK', 'Summery-Performance-PKatt', 'Summery-Performance-Sh', 'Summery-Performance-SoT', 'Summery-Performance-CrdY', 'Summery-Performance-CrdR', 'Summery-Performance-Touches', 'Summery-Performance-Press', 'Summery-Performance-Tkl', 'Summery-Performance-Int',
                   'Summery-Performance-Blocks', 'Summery-Expected-xG', 'Summery-Expected-npxG', 'Summery-Expected-xA', 'Summery-SCA-SCA', 'Summery-SCA-GCA', 'Summery-Passes-Cmp', 'Summery-Passes-Att', 'Summery-Passes-Cmp%', 'Summery-Passes-Prog', 'Summery-Carries-Carries', 'Summery-Carries-Prog', 'Summery-Dribbles-Succ', 'Summery-Dribbles-Att']

columns_passing = ['Passing-Total-Cmp', 'Passing-Total-Att', 'Passing-Total-Cmp%', 'Passing-Total-TotDist', 'Passing-Total-PrgDist', 'Passing-Short-Cmp', 'Passing-Short-Att', 'Passing-Short-Cmp%', 'Passing-Medium-Cmp',
                   'Passing-Medium-Att', 'Passing-Medium-Cmp%', 'Passing-Long-Cmp', 'Passing-Long-Att', 'Passing-Long-Cmp%', 'Passing-Ast', 'Passing-xA', 'Passing-KP', 'Passing-1/3', 'Passing-PPA', 'Passing-CrsPA', 'Passing-Prog']


columns_passing_types = ['PassingTypes-PassAttempted', 'PassingTypes-PassTypes-Live', 'PassingTypes-PassTypes-Dead', 'PassingTypes-PassTypes-FK', 'PassingTypes-PassTypes-TB', 'PassingTypes-PassTypes-Press', 'PassingTypes-PassTypes-Sw', 'PassingTypes-PassTypes-Crs', 'PassingTypes-PassTypes-CK', 'PassingTypes-CornerKicks-In', 'PassingTypes-CornerKicks-Out', 'PassingTypes-CornerKicks-Str',
                         'PassingTypes-Height-Ground', 'PassingTypes-Height-Low', 'PassingTypes-Height-High', 'PassingTypes-BodyParts-Left', 'PassingTypes-BodyParts-Right', 'PassingTypes-BodyParts-Head', 'PassingTypes-BodyParts-TI', 'PassingTypes-BodyParts-Other', 'PassingTypes-Outcomes-Cmp', 'PassingTypes-Outcomes-Off', 'PassingTypes-Outcomes-Out', 'PassingTypes-Outcomes-Int', 'PassingTypes-Outcomes-Blocks']

columns_gca = ['GCA-SCATypes-SCA', 'GCA-SCATypes-PassLive', 'GCA-SCATypes-PassDead', 'GCA-SCATypes-Drib', 'GCA-SCATypes-Sh', 'GCA-SCATypes-Fld', 'GCA-SCATypes-Def',
               'GCA-GCATypes-GCA', 'GCA-GCATypes-PassLive', 'GCA-GCATypes-PassDead', 'GCA-GCATypes-Drib', 'GCA-GCATypes-Sh', 'GCA-GCATypes-Fld', 'GCA-GCATypes-Def']


columns_defense = ['Defence-Tackles-Tkl', 'Defence-Tackles-TklW', 'Defence-Tackles-Def-3rd', 'Defence-Tackles-Mid-3rd', 'Defence-Tackles-Att-3rd', 'Defence-VsDribbles-Tkl', 'Defence-VsDribbles-Att', 'Defence-VsDribbles-Tkl%', 'Defence-VsDribbles-Past', 'Defence-Pressures-Press',
                   'Defence-Pressures-Succ', 'Defence-Pressures-%', 'Defence-Pressures-Def-3rd', 'Defence-Pressures-Mid-3rd', 'Defence-Pressures-Att-3rd', 'Defence-Blocks-Blocks', 'Defence-Blocks-Sh', 'Defence-Blocks-ShSv', 'Defence-Blocks-Pass', 'Defence-Int', 'Defence-Tkl+Int', 'Defence-Clr', 'Defence-Err']


columns_possession = ['Possession-Touches-Touches', 'Possession-Touches-Def-Pen', 'Possession-Touches-Def-3rd', 'Possession-Touches-Mid-3rd', 'Possession-Touches-Att-3rd', 'Possession-Touches-Att-Pen', 'Possession-Touches-Live', 'Possession-Dribbles-Succ', 'Possession-Dribbles-Att', 'Possession-Dribbles-Succ%', 'Possession-Dribbles-#Pl', 'Possession-Dribbles-Megs',
                      'Possession-Carries-Carries', 'Possession-Carries-TotDist', 'Possession-Carries-PrgDist', 'Possession-Carries-Prog', 'Possession-Carries-1/3', 'Possession-Carries-CPA', 'Possession-Carries-Mis', 'Possession-Carries-Dis', 'Possession-Receiving-Targ', 'Possession-Receiving-Rec', 'Possession-Receiving-Rec%', 'Possession-Receiving-Prog']

columns_misc = ['Misc-Performance-CrdY', 'Misc-Performance-CrdR', 'Misc-Performance-2CrdY', 'Misc-Performance-Fls', 'Misc-Performance-Fld', 'Misc-Performance-Off', 'Misc-Performance-Crs', 'Misc-Performance-Int',
                'Misc-Performance-TklW', 'Misc-Performance-PKwon', 'Misc-Performance-PKcon', 'Misc-Performance-OG', 'Misc-Performance-Recov', 'Misc-AerialDuels-Won', 'Misc-AerialDuels-Lost', 'Misc-AerialDuels-Won%']

columns_summary_short = ['Summery-Performance-Gls', 'Summery-Performance-Ast', 'Summery-Performance-PK', 'Summery-Performance-PKatt', 'Summery-Performance-Sh', 'Summery-Performance-SoT', 'Summery-Performance-CrdY',
                         'Summery-Performance-CrdR', 'Misc-Performance-Fls', 'Misc-Performance-Fld', 'Misc-Performance-Off', 'Misc-Performance-Crs', 'Misc-Performance-TklW', 'Misc-Performance-Int', 'Misc-Performance-OG', 'Misc-Performance-PKwon', 'Misc-Performance-PKcon']


columns_keeper = ['KeeperPerformance-SoTA', 'KeeperPerformance-GA', 'KeeperPerformance-Saves', 'KeeperPerformance-Save%', 'KeeperPerformance-CS', 'KeeperPerformance-PSxG', 'KeeperPenaltyKicks-PKatt', 'KeeperPenaltyKicks-PKA', 'KeeperPenaltyKicks-PKsv', 'KeeperPenaltyKicks-PKm', 'KeeperLaunched-Cmp',
                  'KeeperLaunched-Att', 'KeeperLaunched-Cmp%', 'KeeperPasses-Att', 'KeeperPasses-Thr', 'KeeperPasses-Launch%', 'KeeperPasses-AvgLen', 'KeeperGoalKicks-Att', 'KeeperGoalKicks-Launch%', 'KeeperGoalKicks-AvgLen', 'KeeperCrosses-Opp', 'KeeperCrosses-Stp', 'KeeperCrosses-Stp%', 'KeeperSweeper-#OPA', 'KeeperSweeper-AvgDist']

columns_keeper_short = ['KeeperPerformance-SoTA', 'KeeperPerformance-GA', 'KeeperPerformance-Saves', 'KeeperPerformance-Save%',
                        'KeeperPerformance-CS', 'KeeperPenaltyKicks-PKatt', 'KeeperPenaltyKicks-PKA', 'KeeperPenaltyKicks-PKsv', 'KeeperPenaltyKicks-PKm']


def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
        .encode('ascii', 'ignore')\
        .decode("utf-8")

    return str(text).upper()

def browser_driver():

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    chrome_service = Service(BROWSER_DRIVER_PATH)

    return webdriver.Chrome(service=chrome_service, options=options)

if __name__ == "__main__":
    print(PROJECT_DIR, PLAYER_LOG_PREFIX)

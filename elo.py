import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd

from constants import *

# log to stdout
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def load_elo():

    ELO = {}

    for f in os.listdir(os.path.join(DATASETS_DIR, 'elo')):
        team = f.split('.')[0]

        # if team in inv_elomapping:
        #     team = inv_elomapping[team]

        df = pd.read_csv(os.path.join(DATASETS_DIR, 'elo', f))

        df['From'] = pd.to_datetime(df['From'])
        df['To'] = pd.to_datetime(df['To'])
        # we only need data after 2018
        dp = [2018, 1, 1]

        # hist = hist[(hist['Date'].dt.date < datetime.date(*dp)) \
        #             & (hist['PlayerUrl'] == player_link) \
        #             & (hist['Comp'] == league)]

        # hist = hist[(hist['Date'].dt.date < datetime.date(*dp)) \
        #             & (hist['PlayerUrl'] == player_link)]

        df = df[(df['To'].dt.date > datetime.date(*dp))]

        ELO[team] = df

    return ELO


def get_elo(ELO, team_name, date):
    df = ELO[team_name]

    df = df[(df['From'] < date) & (df['To'] < date)]

    try:
        return df.tail(1)['Elo'].values[0]
    except:
        print(team_name, df)
        exit()


if __name__ == "__main__":
    ELO = load_elo()

    leagues = ['EPL', 'SLL', 'ISA']
    seasons = ['2018-2019', '2019-2020', '2020-2021']

    for l in leagues:
        for s in seasons:
            match_history_csv = os.path.join(
                DATASETS_DIR, s + l + 'matches.csv')

            matches = pd.read_csv(match_history_csv)

            matches = matches[['date', 'match_link', 'home', 'away']]

            matches['date'] = pd.to_datetime(matches['date'])

            # print(matches.columns)
            # print(matches.dtypes)

            home_elo = []
            away_elo = []

            for i, row in matches.iterrows():
                # print(i, row['match_link'])

                home_team = "".join(row['home'].split())
                away_team = "".join(row['away'].split())

                if home_team in elomapping:
                    home_team = elomapping[home_team]
                if away_team in elomapping:
                    away_team = elomapping[away_team]

                # print(home_team, away_team, type(row['date']))

                home_elo.append(get_elo(ELO, home_team, row['date']))

                away_elo.append(get_elo(ELO, away_team, row['date']))

                # print(home_elo, away_elo)

            matches['home_elo'] = home_elo
            matches['away_elo'] = away_elo

            filename = os.path.join(DATASETS_DIR, s + l + 'matches_elo.csv')

            with open(filename, 'w') as f:
                matches.to_csv(f, header=True, index=False)

            logging.info("save matches elo to {}".format(filename))

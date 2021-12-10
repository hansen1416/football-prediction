import collections
import sys
import os
import string

import pandas as pd
import numpy as np
from pandas.core import base

from constants import *


def players_data():

    cap = list(map(lambda x: x.upper(), string.ascii_uppercase))
    data = None

    for c in cap:
        df = pd.read_csv(os.path.join(
            DATASETS_DIR, 'player_log_' + c + '.csv'))

        if data is None:
            data = df
        else:
            data = data.append(df)

    data['Date'] = pd.to_datetime(data['Date'])

    return data.reset_index(drop=True)


if __name__ == "__main__":

    # df = players_data()

    # comps = df['Comp'].unique()
    # # print(df)

    # cols = df.columns
    # comp_nan = []

    # for comp in comps:
    #     tmp_df = df[df['Comp'] == comp]

    #     naperc = tmp_df.isna().sum().sum() / tmp_df.shape[0]

    #     comp_nan.append(
    #         {'comp': comp, 'shape': tmp_df.shape[0], 'naperc': naperc})

    #     print("Comp {}, shape {}, na percentage {}".format(
    #         comp, tmp_df.shape[0], naperc))
    #     # print(tmp_df.shape)
    #     # print(tmp_df.isna().sum().sum())

    # comp_nan = sorted(comp_nan, key=lambda x: x['naperc'])

    # print(comp_nan)

    # # print(df['Comp'].unique())

    c = [{'comp': 'Serie A', 'shape': 48494, 'naperc': 5.90689569843692}, {'comp': 'La Liga', 'shape': 51200, 'naperc': 7.20720703125}, {'comp': 'FIFA World Cup', 'shape': 941, 'naperc': 15.479277364505846}, {'comp': 'Ligue 1', 'shape': 8991, 'naperc': 33.6455344233122}, {'comp': 'Premier League', 'shape': 57155, 'naperc': 46.42839646575103}, {'comp': 'Bundesliga', 'shape': 6359, 'naperc': 54.145463123132565}, {'comp': 'Champions Lg', 'shape': 10327, 'naperc': 55.896000774668344}, {'comp': 'UEFA Euro', 'shape': 1136, 'naperc': 56.10299295774648}, {'comp': 'Europa Lg', 'shape': 10704, 'naperc': 64.99598281016442}, {'comp': 'Copa América', 'shape': 534, 'naperc': 80.85205992509363}, {'comp': 'MLS', 'shape': 1231, 'naperc': 109.54183590576767}, {'comp': 'Super Cup', 'shape': 109, 'naperc': 125.36697247706422}, {'comp': 'Supercoppa Italiana', 'shape': 136, 'naperc': 126.6029411764706}, {'comp': 'Asian Cup', 'shape': 65, 'naperc': 126.70769230769231}, {'comp': 'Euro Qualifying', 'shape': 2676, 'naperc': 127.9050822122571}, {'comp': 'Trophée des Champions', 'shape': 47, 'naperc': 128.0}, {'comp': 'EFL Cup', 'shape': 4314, 'naperc': 128.25776541492814}, {'comp': 'Supercopa de España', 'shape': 331, 'naperc': 128.71601208459214}, {'comp': 'Community Shield', 'shape': 163, 'naperc': 129.38036809815952}, {'comp': 'FIFA Confederations Cup', 'shape': 73, 'naperc': 130.32876712328766}, {'comp': 'DFB-Pokal', 'shape': 571, 'naperc': 130.55866900175133}, {'comp': 'Süper Lig', 'shape': 4444, 'naperc': 131.07358235823583}, {'comp': 'Coupe de France', 'shape': 713, 'naperc': 131.38849929873774}, {'comp': 'Copa del Rey', 'shape': 6208, 'naperc': 131.67380798969072}, {'comp': 'Copa América Centenario', 'shape': 115, 'naperc': 131.8}, {'comp': 'UEFA Nations League', 'shape': 2612, 'naperc': 131.9552067381317}, {'comp': 'Libertadores', 'shape': 587, 'naperc': 132.23850085178876}, {'comp': 'FA Cup', 'shape': 5964, 'naperc': 132.47551978537894}, {'comp': 'Coppa Italia', 'shape': 4044, 'naperc': 132.48862512363996}, {'comp': 'Premiership', 'shape': 1356, 'naperc': 133.1976401179941}, {'comp': 'Dutch Eredivisie', 'shape': 4678, 'naperc': 133.28858486532707}, {'comp': 'Super League', 'shape': 1952, 'naperc': 133.60655737704917}, {'comp': 'Championship', 'shape': 30392, 'naperc': 133.9577191366149}, {'comp': 'Swiss Super League', 'shape': 1092, 'naperc': 134.25824175824175}, {'comp': 'Liga MX', 'shape': 1201, 'naperc': 134.89508742714403}, {'comp': 'Segunda División', 'shape': 27273, 'naperc': 135.49818501814983}, {'comp': 'WCQ — UEFA (M)', 'shape': 2486, 'naperc': 136.20233306516494}, {'comp': 'WCQ', 'shape': 926, 'naperc': 136.23974082073434}, {'comp': 'League One', 'shape': 3329, 'naperc': 136.67347551817363}, {'comp': 'Coupe de la Ligue', 'shape': 475, 'naperc': 137.0757894736842}, {'comp': 'Tippeligaen', 'shape': 87, 'naperc': 138.75862068965517}, {'comp': 'National League', 'shape': 169, 'naperc': 139.3846153846154}, {'comp': 'Friendlies (M)', 'shape': 6860, 'naperc': 139.78804664723032}, {'comp': 'PL2 — Div. 2', 'shape': 1547, 'naperc': 140.26308985132513}, {'comp': 'PL2 — Div. 1', 'shape': 2598, 'naperc': 140.8791377983064}, {'comp': 'WCQ — AFC (M)', 'shape': 221, 'naperc': 141.46153846153845}, {
        'comp': 'League Cup', 'shape': 772, 'naperc': 142.66709844559585}, {'comp': 'Gold Cup', 'shape': 115, 'naperc': 128.9304347826087}, {'comp': 'Africa Cup of Nations', 'shape': 545, 'naperc': 129.0697247706422}, {'comp': 'Primeira Liga', 'shape': 3768, 'naperc': 130.96523354564755}, {'comp': 'Série A', 'shape': 1590, 'naperc': 131.78490566037735}, {'comp': 'Copa Sudamericana', 'shape': 37, 'naperc': 132.02702702702703}, {'comp': 'DFL-Supercup', 'shape': 38, 'naperc': 132.07894736842104}, {'comp': 'Eliteserien', 'shape': 111, 'naperc': 132.26126126126127}, {'comp': 'Sudamericana', 'shape': 290, 'naperc': 132.44137931034481}, {'comp': 'Primera Div', 'shape': 2208, 'naperc': 133.15715579710144}, {'comp': 'Allsvenskan', 'shape': 360, 'naperc': 133.16666666666666}, {'comp': 'Copa Libertadores', 'shape': 66, 'naperc': 133.5}, {'comp': 'J1 League', 'shape': 108, 'naperc': 133.7037037037037}, {'comp': 'USL Champ', 'shape': 20, 'naperc': 133.75}, {'comp': 'UEL Play-offs', 'shape': 12, 'naperc': 133.83333333333334}, {'comp': 'Superliga', 'shape': 910, 'naperc': 133.84835164835164}, {'comp': 'USL Championship', 'shape': 6, 'naperc': 134.33333333333334}, {'comp': 'Série B', 'shape': 91, 'naperc': 134.43956043956044}, {'comp': '3. Liga', 'shape': 188, 'naperc': 134.6595744680851}, {'comp': '2. Bundesliga', 'shape': 791, 'naperc': 134.71554993678888}, {'comp': 'Eerste Divisie', 'shape': 613, 'naperc': 134.8042414355628}, {'comp': 'Serie B', 'shape': 15073, 'naperc': 134.8264446361043}, {'comp': 'A-League', 'shape': 227, 'naperc': 134.94273127753303}, {'comp': 'First Division A', 'shape': 3178, 'naperc': 135.56356198867212}, {'comp': 'League Two', 'shape': 1053, 'naperc': 136.78252611585944}, {'comp': 'Ligue 2', 'shape': 940, 'naperc': 137.15}, {'comp': 'Ekstraklasa', 'shape': 871, 'naperc': 137.45120551090702}, {'comp': 'Pro League', 'shape': 1020, 'naperc': 138.03333333333333}, {'comp': '1. HNL', 'shape': 715, 'naperc': 138.36503496503497}, {'comp': 'SuperLiga', 'shape': 507, 'naperc': 138.41617357001974}, {'comp': 'First League', 'shape': 824, 'naperc': 138.71116504854368}, {'comp': 'Liga I', 'shape': 965, 'naperc': 139.49844559585492}, {'comp': 'Africa Cup of Nations qualification', 'shape': 1075, 'naperc': 139.84093023255815}, {'comp': 'Rel/Pro Play-offs', 'shape': 7, 'naperc': 141.0}, {'comp': 'WCQ — CAF (M)', 'shape': 429, 'naperc': 141.34032634032633}, {'comp': 'First Division B', 'shape': 57, 'naperc': 142.33333333333334}, {'comp': 'WCQ — OFC (M)', 'shape': 6, 'naperc': 142.33333333333334}, {'comp': 'K League 1', 'shape': 53, 'naperc': 142.41509433962264}, {'comp': 'K-League', 'shape': 48, 'naperc': 142.4375}, {'comp': 'Premier Division', 'shape': 66, 'naperc': 142.53030303030303}, {'comp': 'Primera División', 'shape': 301, 'naperc': 142.64451827242524}, {'comp': 'OFC Nations Cup', 'shape': 1, 'naperc': 143.0}, {'comp': 'Superettan', 'shape': 15, 'naperc': 143.0}, {'comp': 'NB I', 'shape': 15, 'naperc': 143.0}, {'comp': 'J2 League', 'shape': 23, 'naperc': 143.0}, {'comp': 'U19 Bundesliga', 'shape': 120, 'naperc': 143.06666666666666}, {'comp': 'U17 Bundesliga', 'shape': 30, 'naperc': 143.26666666666668}]

    ls = ['Serie A',
          'La Liga',
          'FIFA World Cup',
          'Ligue 1',
          'Premier League',
          'Bundesliga',
          'Champions Lg',
          'UEFA Euro',
          'Europa Lg', ]

    st = 0
    lst = 0
    xst = 0

    for i in c:
        if i['comp'] in ls:
            lst += i['shape']
        else:
            xst += i['shape']

        st += i['shape']

    print(st, lst, xst)

import sys
import os

import pandas as pd
import numpy as np

from constants import *


def load_df(weights, weight_name):
    data = None

    base_cols = ['league', 'match_link', 'Season',
                 'date', 'score', 'label', 'spread']

    for league in ['EPL', 'ISA', 'SLL']:
        for season in ['2018-2019', '2019-2020', '2020-2021']:
            f = season + league + '-' + str(HISTORY_LENGTH) + '.csv'

            df = pd.read_csv(os.path.join(DATASETS_DIR, f))

            num_cols = list(set(df.columns).difference(set(base_cols)))

            new_df = pd.DataFrame(columns=df.columns)
            tmp_num_df = pd.DataFrame(columns=num_cols)

            # print(tmp_df)
            # exit()

            tmp_match_link = None

            for _, row in df.iterrows():

                row_df = pd.DataFrame([row])

                if row['match_link'] == tmp_match_link:

                    tmp_num_df = tmp_num_df.append(
                        row_df[num_cols] * weights[int(row_df['history_i'].values[0])])

                    # print(tmp_num_df.sum(axis=0))
                    # exit()

                else:

                    # print(new_df['match_link'],
                    #       new_df['score'], new_df['label'], new_df)
                    # break

                    tmp_match_link = row['match_link']
                    tmp_num_df = row_df[num_cols] * \
                        weights[int(row_df['history_i'].values[0])]

                if tmp_num_df.shape[0] == 10:

                    tmp_num_df = tmp_num_df.sum(axis=0)

                    for col in base_cols:
                        # print(row_df[col])

                        tmp_num_df[col] = row_df[col].values[0]

                    new_df = new_df.append(pd.DataFrame([tmp_num_df]))

                    # print(tmp_num_df, tmp_num_df['away_Passing-Long-Att'])

            new_df_f = os.path.join(
                DATASETS_DIR, 'weighted', season + league + 'weighted' + weight_name + '.csv')

            new_df = new_df.drop(columns=['history_i'])
            new_df.to_csv(new_df_f, index=False)

            print("file saved to {}".format(new_df_f))


if __name__ == "__main__":

    for i, w in MATCH_WEIGHTS.items():
        load_df(w, str(i))

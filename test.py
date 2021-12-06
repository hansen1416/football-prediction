import sys
import os

import pandas as pd
import numpy as np
from pandas.core import base

from constants import *


if __name__ == "__main__":

    for league in ['EPL', 'ISA', 'SLL']:
        for season in ['2018-2019', '2019-2020', '2020-2021']:
            for i in range(3):
                new_df_f = os.path.join(
                    DATASETS_DIR, season + league + 'weighted' + str(i) + '.csv')

                df = pd.read_csv(new_df_f)

                print(df.shape)

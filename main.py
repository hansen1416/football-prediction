import argparse

import classification
import plots

desc = 'The is a soccer result prediction program.\n run'

# Initiate the parser with a description
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--plot', action='store', type=str,
                    help="'--plot all' generate a char of all classification matrics;\
                         '--plot league' generate a chart that compare accuracy score and f1 score of different leagues;\
                         '--plot elo' compares leagues with and without ELO;\
                         '--league-elo league' compares results with and without ELO'")

args = parser.parse_args()

if __name__ == "__main__":

    metrics_table = classification.main()

    metrics_table_num = metrics_table.copy()

    for l1, v1 in metrics_table_num.items():
        for l2, v2 in v1.items():
            for l3, v3 in v2.items():
                metrics_table_num[l1][l2][l3] = float(v3)

    if args.plot == 'all':
        plots.big_table(metrics_table)
    elif args.plot == 'league':
        plots.league_comparison(metrics_table_num)
    elif args.plot == 'elo':
        plots.elo_comparison(metrics_table_num)
    elif args.plot == 'league-elo':
        plots.league_elo_comparison(metrics_table_num)

import pandas as pd
import numpy as np
#import openpyxl
import os
from tqdm import tqdm
tqdm.pandas()
import csv

def count_games(df):
    df.index = df.index.str.split('-').str[0]

    games = df.index.unique()

    games = games.tolist()

    return len(games)

def count_wins(df):
    
    # remove rows that have same index
    df = df[~df.index.duplicated(keep='first')]

    games = df.index.unique()

    games = games.tolist()

    num_wins_white = 0
    num_wins_black = 0

    #print(games[0])
    #print(df.loc[games[0]])

    for game in games:
        game_df = df.loc[game]
        if int(game_df['Result']) == 0:
            num_wins_white += 1
        elif int(game_df['Result']) == 1:
            num_wins_black += 1

    return num_wins_white, num_wins_black

def main():

    dict_games = {}

    BASE_FOLDER = '../data/5_lichess_csv_ML_ready'

    for csv_file_name in tqdm(os.listdir(BASE_FOLDER)):
        INPUT_FILE_PATH = f'{BASE_FOLDER}/{csv_file_name}'
        if not csv_file_name.endswith('.pkl'):
            continue
        df = pd.read_pickle(INPUT_FILE_PATH)

        print(csv_file_name)
        dict_games[csv_file_name] = count_games(df), count_wins(df)

    #print(dict_games)

    with open('../data/num_wins.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'num_games', 'num_wins_white', 'num_wins_black', 'white_win_rate', 'black_win_rate'])
        # order by elo rating
        dict_games = dict(sorted(dict_games.items(), key=lambda item: int(item[0].split('_')[1])))
        for key, value in dict_games.items():
            writer.writerow([key, value[0], value[1][0], value[1][1], round(value[1][0]/value[0],2), round(value[1][1]/value[0],2)])

if __name__ == "__main__":
    main()


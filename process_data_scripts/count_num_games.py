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
    print(len(games))

    return len(games)

def main():

    dict_games = {}

    BASE_FOLDER = '../data/5_lichess_csv_ML_ready'

    for csv_file_name in tqdm(os.listdir(BASE_FOLDER)):
        INPUT_FILE_PATH = f'{BASE_FOLDER}/{csv_file_name}'
        if not csv_file_name.endswith('.pkl'):
            continue
        df = pd.read_pickle(INPUT_FILE_PATH)

        print(csv_file_name)
        dict_games[csv_file_name] = count_games(df), df.shape[0]

    with open('../data/num_games.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'num_games', 'num_rows'])
        for key, value in dict_games.items():
            writer.writerow([key, value[0], value[1]])

if __name__ == "__main__":
    main()


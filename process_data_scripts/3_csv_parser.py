import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import csv

'''
    This script removes irrelevant fields in the CSV files and converts the winner to an integer
'''


def clean_pgn_mainline(pgn_mainline: str) -> str:

    '''
    This function cleans the PGN mainline
    Parameters:
        pgn_mainline (str): The PGN mainline
    Returns:
        pgn_mainline (str): The cleaned PGN mainline
    '''

    if(type(pgn_mainline) != str):
        print(f"{pgn_mainline} of type {type(pgn_mainline)}")
        return ""
    else:
        return pgn_mainline

#! NOT USED
def replace_question_mark_with_na(value: str) -> str:
    if '?' in value:
        return 0
    else:
        return int(value)

def replace_winner_by_int(value: str) -> int:

    '''
    This function replaces the winner by an integer
    Parameters:
        value (str): The winner
    Returns:
        winner (int): 0 if white won, 1 if black won, 2 if draw
    '''

    if value == '1-0':
        return 0
    if value == '0-1':
        return 1
    # value == '1/2-1/2'
    else:
        return 2   

def opening_stats(row: pd.Series,opening_dict:dict):

    '''

    This function calculates the wins, losses and draws for each opening
    Parameters:
        row (pd.Series): The row of the CSV file
        opening_dict (dict): The dictionary with the openings and their stats
    Returns:
        None
    
    '''
    
    if row['Opening'] not in opening_dict:
        opening_dict[row['Opening']] = {
            'Wins': 0,
            'Losses': 0,
            'Draws': 0
        }
    
    if row['Result'] == 1:
        opening_dict[row['Opening']]['Wins'] += 1
    elif row['Result'] == 0:
        opening_dict[row['Opening']]['Losses'] += 1
    else:
        opening_dict[row['Opening']]['Draws'] += 1    

def main():

    '''
    This function iterates over all CSV files and removes irrelevant fields and converts the winner to an integer
    '''

    BASE_FOLDER = '../data/2_lichess_csv_filtered'
    OUTPUT_FOLDER = '../data/3_lichess_csv_parsed'
    OPENING_OUTPUT_FOLDER = '../data/opening_stats'

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(OPENING_OUTPUT_FOLDER):
        os.makedirs(OPENING_OUTPUT_FOLDER)

    for csv_file_name in tqdm(os.listdir(BASE_FOLDER)):

        print(f"Reading file {csv_file_name}")

        INPUT_CSV = f'{BASE_FOLDER}/{csv_file_name}'
        
        df = pd.read_csv(INPUT_CSV, index_col=1, header=0)
        #print(df.info())
        
        df.drop_duplicates(subset=None, keep='first', inplace=True)
        # not interested in 'Time forfeit' games
        df = df[df['Termination'] == 'Normal']

        # if all games are from the same type of event, no need to keep it
        df.drop(columns=['Termination', 'Event'], inplace=True)

        # df['WhiteElo'] = df['WhiteElo'].apply(replace_question_mark_with_na)
        # df['WhiteElo'] = df['WhiteElo'].astype('int16')
        # df['BlackElo'] = df['BlackElo'].apply(replace_question_mark_with_na)
        # df['BlackElo'] = df['BlackElo'].astype('int16')
        # df = df[df['BlackElo'] != 0]
        # df = df[df['WhiteElo'] != 0]
        # print(sorted(list(df['WhiteElo'].unique())))
        # print(sorted(list(df['BlackElo'].unique())))

        df['Result'] = df['Result'].apply(replace_winner_by_int)
        df['Result'] = df['Result'].astype('int16')

        # identify rows with float values in the column, by mistake
        moves_mask = df['Moves'].apply(lambda x: isinstance(x, float))
        # drop rows with float values in the column
        df = df[~moves_mask]
        
        df['Moves'] = df['Moves'].apply(clean_pgn_mainline)

        #print(df.info())
        # print(sum(df['Moves'].str.len() == np.nan))
        # print(sorted(list(set(df['Moves'].str.len()))))

        #csv_file_name = csv_file_name.replace('.csv', '_parsed.csv')

        #! Saving opening wins,losses and draws
        opening_dict = {}
        df.apply(opening_stats, axis=1, args=(opening_dict,))

        OUTPUT_OPENING = f'{OPENING_OUTPUT_FOLDER}/{csv_file_name}'
        with open(OUTPUT_OPENING, 'w') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames = ['Opening', 'Wins', 'Losses', 'Draws', 'Total', 'Winrate']) 
            writer.writeheader() 
            for key in opening_dict.keys():
                total = opening_dict[key]['Wins'] + opening_dict[key]['Losses'] + opening_dict[key]['Draws']
                winrate = opening_dict[key]['Wins'] / total
                writer.writerow({'Opening': key, 'Wins': opening_dict[key]['Wins'], 'Losses': opening_dict[key]['Losses'], 'Draws': opening_dict[key]['Draws'], 'Total': total, 'Winrate': winrate})


        OUTPUT_CSV = f'{OUTPUT_FOLDER}/{csv_file_name}'

        print(f"Writing file {OUTPUT_CSV}")
        df.to_csv(OUTPUT_CSV)

    
if __name__ == "__main__":
    main()


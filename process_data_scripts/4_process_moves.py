import pandas as pd
import chess.pgn
import io
import swifter
import os
from tqdm import tqdm

'''
    This script converts a CSV file to a CSV file with one row per move

    Example:
    Input:
    Id,WhiteElo,BlackElo,Result,Moves

    Output:
    Id,MoveNumber,WhiteElo,BlackElo,Result,Board (FEN) (for each move)
'''

# define custom function to transform each row into one row per move available from the PGN
def process_row(row: pd.Series):

    '''
    This function transforms a row into one row per move available from the PGN
    Parameters:
        row (pd.Series): The row to transform
    Returns:
        result (pd.DataFrame): The transformed row
    '''
    moves = row['Moves']
    dict_result = {'Id': [], 'MoveNumber': [], 'WhiteElo': [], 'BlackElo': [], 'Result': [], 'Board':[]}

    #convert moves to string
    moves = str(moves)
    pgn = chess.pgn.read_game(io.StringIO(moves))
    board = pgn.board()

    move_number = 0
    for move in pgn.mainline_moves():
        if move_number > 5:
            dict_result['Id'].append(f"{row['Site']}-{move_number}")
            dict_result['MoveNumber'].append(move_number)
            dict_result['WhiteElo'].append(row['WhiteElo'])
            dict_result['BlackElo'].append(row['BlackElo'])
            dict_result['Result'].append(row['Result'])
            dict_result['Board'].append(board.fen()) 
        move_number += 1

        board.push(move)
        # print(board.fen(), move)
    return pd.DataFrame(dict_result)

def main():

    '''
    This function iterates over all CSV files and converts them to CSV files with one row per move available from the PGN

    It uses the swifter library to parallelize the processing of each row in the dataframe and concatenate the results into a single dataframe
    '''
    BASE_FOLDER = '../data/3_lichess_csv_parsed'
    OUTPUT_FOLDER = '../data/4_lichess_csv_processed'

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for csv_file_name in tqdm(os.listdir(BASE_FOLDER)):

        print(f"Reading file {csv_file_name}")

        INPUT_FILE_PATH = f'{BASE_FOLDER}/{csv_file_name}'

        df = pd.read_csv(INPUT_FILE_PATH, index_col=0, header=0)

        # mask = df.index.duplicated(keep=False)
        # duplicates = df.index[mask].tolist()
        # print(len(duplicates))
        # print(len(mask))
        # exit(0)

        print('Processing each game and transforming into "expanded" dataset')
        # apply custom function to each row in dataframe and concatenate results
        #print(df.head())
        df.reset_index(drop=False, inplace=True)
        df.index.name = 'Id'
        print(df.head())

        result = pd.concat(df.swifter.apply(process_row, axis=1).tolist(), ignore_index=True).set_index('Id')
        print(result.head())
        
        OUTPUT_FILE = f'{OUTPUT_FOLDER}/{csv_file_name}'
        print(f"Saving file {OUTPUT_FILE}")
        result.to_csv(OUTPUT_FILE)
        

if __name__ == "__main__":
    main()
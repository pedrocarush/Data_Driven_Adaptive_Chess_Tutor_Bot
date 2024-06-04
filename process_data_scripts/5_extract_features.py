import pandas as pd
import chess
import io
from tqdm import tqdm
tqdm.pandas()
import swifter
import numpy as np
import os

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_extraction

'''
    This script extracts features from the board and adds them to the dataframe as columns, which is then saved to a CSV file
    This is later used to train the models
    
'''


def extract_features_from_board(row: pd.Series) -> pd.Series:

    ''' 
    This function extracts features from the board and adds them to the dataframe as columns
    Parameters:
        row (pd.Series): The row to transform
    Returns:
        row (pd.Series): The transformed row
    '''

    fen_board = row['Board']
    
    board = chess.Board(fen_board)

    feature_extraction._init_square_data(board)
    
    pieces_dict = feature_extraction.count_pieces(fen_board)

    #piece_lists_dict = piece_lists(board)
    
    #sliding_pieces_mobility_dict = sliding_pieces_mobility(board)
    
    #attack_and_defend_maps_dict = attack_and_defend_maps(board)
    
    mobility_dict = feature_extraction.mobility(board)

    pawn_structure_dict = feature_extraction.pawn_structure(board)

    king_safety_dict = feature_extraction.king_safety(board)

    trapped_pieces_dict = feature_extraction.trapped_pieces(board)

    check_dict = feature_extraction.check(board)

    merged_dict = pieces_dict | mobility_dict | pawn_structure_dict | king_safety_dict | trapped_pieces_dict | check_dict
    row.update(merged_dict)
    
    return row

def initialize_dataframe(df):

    '''
    This function initializes the dataframe with the columns that will be used to store the features extracted from the board
    Parameters:
        df (pd.DataFrame): The dataframe to initialize
    Returns:
        df (pd.DataFrame): The initialized dataframe
    '''

    df = df.reindex(df.columns.tolist() + [
        #Piece count
        *[f'{p}_white' for p in ('K', 'Q', 'R', 'B', 'N', 'P')],
        *[f'{p}_black' for p in ('k', 'q', 'r', 'b', 'n', 'p')],
        #Mobility
        'Mobility_white', 'Mobility_black',
        #Pawn structure
        'Passed_pawns_white', 'Isolated_pawns_white', 'Doubled_pawns_white', 'Passed_pawns_black', 'Isolated_pawns_black', 'Doubled_pawns_black',
        #King safety
        'King_pawn_shield_white', 'King_pawn_shield_black', 'King_zone_attacked_squares_white', 'King_zone_attacked_squares_black', 'King_zone_controlled_squares_white', 'King_zone_controlled_squares_black',
        #Trapped pieces
        'Trapped_bishops_white', 'Trapped_bishops_black', 'Trapped_rooks_white', 'Trapped_rooks_black', 'Trapped_queens_white', 'Trapped_queens_black',
        #Check
        'Check_white', 'Check_black',
    ], axis=1, fill_value=np.int8(-1))    

    return df

def main():

    '''
    This function iterates over all CSV files and extracts features from the board and adds them to the dataframe as columns, which is then saved to a CSV file
    The dataframes are processed in chunks of 100000 rows to avoid memory issues

    The dataframes are then saved as CSV and pickle files
    '''

    BASE_FOLDER = '../data/4_lichess_csv_processed'
    OUTPUT_FOLDER= '../data/5_lichess_csv_ML_ready'

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


    print(os.listdir(BASE_FOLDER))

    for csv_file_name in tqdm(os.listdir(BASE_FOLDER)):
        print(f"Reading file {csv_file_name}")

        #if os.path.exists(f'{OUTPUT_FOLDER}/{csv_file_name}'):
        #    print(f"File {csv_file_name} already exists. Skipping")
        #    continue

        INPUT_FILE_PATH = f'{BASE_FOLDER}/{csv_file_name}'
        OUTPUT_FILE = f'{OUTPUT_FOLDER}/{csv_file_name}'
    
        df = pd.read_csv(INPUT_FILE_PATH, index_col=0, header=0)

        df = initialize_dataframe(df)
        #df.sort_index(inplace=True,axis=1)

        n = 100000  #chunk row size
        
        print("Applying 'extract_features_from_board'")
        for i in range(0,df.shape[0],n):
            print(str(i) + "/" + str(df.shape[0]) + " rows", end='\n')
            df[i:i+n] = df[i:i+n].swifter.allow_dask_on_strings().apply(extract_features_from_board, axis=1)
        print(df.head)
        
        print(f"Saving file {OUTPUT_FILE}")
        df.to_csv(OUTPUT_FILE)
        # pickle
        df.to_pickle(OUTPUT_FILE + '.pkl')

        # fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        # print(count_pieces(fen))

if __name__ == "__main__":
    main()
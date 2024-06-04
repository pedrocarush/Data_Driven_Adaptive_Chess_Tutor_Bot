import chess
import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_extraction

'''
    This file extracts the features of the chess board, the same used to train the models in the process_data_scripts folder

    board_features -> returns a dictionary with all the features extracted from the board

'''

def board_features(board:chess.Board):
    
    feature_extraction._init_square_data(board)
    
    pieces_dict = feature_extraction.count_pieces(board.fen())
    
    mobility_dict = feature_extraction.mobility(board)

    pawn_structure_dict = feature_extraction.pawn_structure(board)

    king_safety_dict = feature_extraction.king_safety(board)

    trapped_pieces_dict = feature_extraction.trapped_pieces(board)

    check_dict = feature_extraction.check(board)

    merged_dict = {"Result":0} | pieces_dict | mobility_dict | pawn_structure_dict | king_safety_dict | trapped_pieces_dict | check_dict

    df = pd.DataFrame(merged_dict, index=[0])

    df.sort_index(inplace=True,axis=1)

    return df

import chess
import pandas as pd
from extract_features import board_features
#import chrome_driver_test
import cohere

feature_explanation = {
    'K_white': 'Number of white kings',
    'Q_white': 'Number of white queens',
    'R_white': 'Number of white rooks',
    'B_white': 'Number of white bishops',
    'N_white': 'Number of white knights',
    'P_white': 'Number of white pawns',
    'k_black': 'Number of black kings',
    'q_black': 'Number of black queens',
    'r_black': 'Number of black rooks',
    'b_black': 'Number of black bishops',
    'n_black': 'Number of black knights',
    'p_black': 'Number of black pawns',
    'Mobility_white': 'Number of legal moves for white',
    'Mobility_black': 'Number of legal moves for black',
    'Passed_pawns_white': 'Number of white passed pawns',
    'Passed_pawns_black': 'Number of black passed pawns',
    'Isolated_pawns_white': 'Number of white isolated pawns',
    'Isolated_pawns_black': 'Number of black isolated pawns',
    'Doubled_pawns_white': 'Number of white doubled pawns',
    'Doubled_pawns_black': 'Number of black doubled pawns',
    'King_pawn_shield_white': 'Number of pawns in the white king\'s pawn shield',
    'King_pawn_shield_black': 'Number of pawns in the black king\'s pawn shield',
    'King_zone_attacked_squares_white': 'Number of squares attacked by black in the white king\'s zone',
    'King_zone_attacked_squares_black': 'Number of squares attacked by white in the black king\'s zone',
    'King_zone_controlled_squares_white': 'Number of squares controlled by white in the white king\'s zone',
    'King_zone_controlled_squares_black': 'Number of squares controlled by black in the black king\'s zone',
    'Trapped_bishops_white': 'Number of trapped white bishops',
    'Trapped_bishops_black': 'Number of trapped black bishops',
    'Trapped_rooks_white': 'Number of trapped white rooks',
    'Trapped_rooks_black': 'Number of trapped black rooks',
    'Trapped_queens_white': 'Number of trapped white queens',
    'Trapped_queens_black': 'Number of trapped black queens',
    'Check_white': 'If white is in check',
    'Check_black': 'If black is in check',
}

#! This needs to be updated with the correct features 
# The positive and negative features are used to determine if a feature is good or bad for a player
# The positive features are good for the white player and bad for the black player
# The negative features are good for the black player and bad for the white player
positive_features = ["K_white", "Q_white", "R_white", "B_white", "N_white", "P_white",
                      "Mobility_white",
                        "Passed_pawns_white","Isolated_pawns_black","Doubled_pawns_black",
                        "King_pawn_shield_white","King_zone_attacked_squares_black","King_zone_controlled_squares_white",
                        "Trapped_bishops_black","Trapped_rooks_black","Trapped_queens_black",
                        "Check_black"]

negative_features = ["k_black", "q_black", "r_black", "b_black", "n_black", "p_black",
                      "Mobility_black",
                        "Passed_pawns_black","Isolated_pawns_white","Doubled_pawns_white",
                        "King_pawn_shield_black","King_zone_attacked_squares_white","King_zone_controlled_squares_black",
                        "Trapped_bishops_white","Trapped_rooks_white","Trapped_queens_white",
                        "Check_white"]

# NOT USED CURRENTLY
extra_preamble = '''
You will also receive information about the best possible move in that situation and the changes in the features of the board if that move was made, to help the student understand the best possible move in that situation, or you will receive information about the best possible counter move for the opponent after the student's move, to help the student understand the consequences of their move.
Always specify the piece moved and the square it moved to when commenting on the best possible move and the best possible counter move.
'''

preamble_template = '''

## Task & Context
You are a chess tutor giving advice to a student. 
You will be making comments on the moves made by the student to help them improve their chess skills. 
The comments will be based on the changes in the features of the board after the move. Don't comment on anything that isn't present in the prompt given.

## Style Guide
Try to keep the comments concise and to the point, while also keeping positive and encouraging.

## Meaning of each Feature

'''
preamble_template += '\n'.join([f'- **{k}**: {v}' for k, v in feature_explanation.items()])

f = open("cohere_key.txt","r")

api_key = f.readline()

co = cohere.Client(api_key)

def opening_comment(player_color,last_board:chess.Board,played_move:str,opening:str,opening_winrate:float,best_opening_move:str,best_opening:str,best_opening_winrate:float):
    
    print("Opening comment\n")

    prompt = "## Instructions\nYou are a chess tutor giving advice to a student. Make a comment on the opening played by the student considering the following:\n## Input\n"
    prompt += "student is playing " + player_color + " side\n"
    prompt += "opening played by the student: " + opening + "\n"
    prompt += "winrate of the opening: " + str(opening_winrate) + "\n"
    prompt += "move played by the student: " + played_move + "\n"
    prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(played_move[:2])).symbol() + "\n"

    prompt += "\n\n"

    
    prompt += "best possible move in this situation: " + best_opening_move + "\n"
    prompt += "best opening: " + best_opening + "\n"
    prompt += "winrate of the best opening: " + str(best_opening_winrate) + "\n"
    prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(best_opening_move[:2])).symbol() + "\n"

    print("prompt")
    print("\n")
    #print(prompt)
    print("\n\n\n\n")


    received_message = co.chat(
            message=prompt,
            
            preamble=preamble_template,
            temperature=0.3
        )
    print(received_message.text)
    return

# The move_comment function will be called after the move has been played and the evaluation has been calculated
# The function will be responsible for printing a comment about the move
def move_comment(player_color,last_board:chess.Board,last_eval:float,current_board:chess.Board,played_move:str,move_eval:float,best_eval:float,best_move:str,best_counter_move:str):
    print("Move comment\n")

    #! Checkmate cases

    # if best eval = 2, then the player could have ended the game with a checkmate
    if best_eval == 2:
        prompt = "## Instructions\nYou are a chess tutor giving advice to a student. Make a comment on the move made by the student considering the following:\n## Input\n"
        prompt += "student missed a checkmate opportunity\n"
        prompt += "student is playing " + player_color + " side\n"
        prompt += "quality of the move: bad\n"

        prompt += "best possible move in this situation that leads to checkmate: " + best_move + "\n"
        prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(best_move[:2])).symbol() + "\n"
        print("prompt")
        print("\n")
        #print(prompt)
        print("\n\n\n\n")

        received_message = co.chat(
            message=prompt,
            
            preamble=preamble_template,
            temperature=0.3
        )
        print(received_message.text)
        return
    # if the move made by the student is -2, then the player opened themselves to a checkmate from the opponent
    elif move_eval == -2:
        prompt = "## Instructions\nYou are a chess tutor giving advice to a student. Make a comment on the move made by the student considering the following:\n## Input\n"
        prompt += "student opened themselves to a checkmate from the opponent \n"
        prompt += "student is playing " + player_color + " side\n"
        prompt += "quality of the move: bad\n"

        prompt += "best possible counter move the opponent could have made after student's move that leads to checkmate: " + best_counter_move + "\n"
        prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(best_counter_move[:2])).symbol() + "\n"
        print("prompt")
        print("\n")
        #print(prompt)
        print("\n\n\n\n")

        received_message = co.chat(
            message=prompt,
            
            preamble=preamble_template,
            temperature=0.3
        )
        print(received_message.text)
        return


    pd.set_option('display.max_columns', 100)

    #! Determine the quality of the move
    # If the move evaluation is close to the best evaluation, the move is considered GOOD
    if move_eval >= best_eval * 0.9:
        print(f"The move {played_move} is a good move\n")
        quality_flag = "good"

        prompt = "## Instructions\nYou are a chess tutor giving advice to a student. Make a comment on the move made by the student considering the following:\n## Input\n"
        prompt += "student is playing " + player_color + " side\n"
        prompt += "quality of the move: good\n"

    # If the move evaluation is better than the last evaluation or the best evaluation is significantly worse than the last evaluation, the move is considered DECENT
    elif move_eval >= last_eval * 0.85 or best_eval <= last_eval * 0.8:
        print(f"The move {played_move} is a decent move\n")
        quality_flag = "decent"

        prompt = "## Instructions\nYou are a chess tutor giving advice to a student. Make a comment on the move made by the student and comment on the best possible move in that situation, considering the following:\n## Input\n"
        prompt += "student is playing " + player_color + " side\n"
        prompt += "quality of the move: decent\n"

    # If the move evaluation is worse than the last evaluation, the move is considered BAD
    else:
        print(f"The move {played_move} is a bad move\n")
        quality_flag = "bad"

        prompt = "## Instructions\nYou are a chess tutor giving advice to a student. Make a comment on the move made by the student and on the best possible counter move for the opponent after the student's move, considering the following:\n## Input\n"
            
        prompt += "student is playing " + player_color + " side\n"
        prompt += "quality of the move: bad\n"

    #! Give the move made by the player
    prompt += "move played by the student:" + played_move + "\n"
    prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(played_move[:2])).symbol() + "\n"


    #! calculate the difference in features between the last board and the current board
    last_board_features = board_features(last_board)
    current_board_features = board_features(current_board)

    #difference in features
    features_diff = current_board_features.subtract(last_board_features)
    #drop the columns that are all 0
    features_diff = features_diff.loc[:, (features_diff != 0).any(axis=0)]

    print(features_diff.head())

    #print the features that changed
    #features_diff = current_board_features.compare(last_board_features)
    #print(features_diff.head())

    prompt += "positive changes in the features of the board after the move:\n"

    for index, row in features_diff.iterrows():
        for column in features_diff.columns:
            if ((row[column] > 0 and column in positive_features) or (row[column] < 0 and column in negative_features)) and player_color == "white":
                prompt += f"\t{column}: {row[column]}\n" 
            elif ((row[column] > 0 and column in negative_features) or (row[column] < 0 and column in positive_features)) and player_color == "black":
                prompt += f"\t{column}: {row[column]}\n"
    if prompt.endswith("positive changes in the features of the board after the move:\n"):
        prompt += "\tNone\n"

    prompt += "negative changes in the features of the board after the move:\n"
    for index, row in features_diff.iterrows():
        for column in features_diff.columns:
            if ((row[column] < 0 and column in positive_features) or (row[column] > 0 and column in negative_features)) and player_color == "white":
                prompt += f"\t{column}: {row[column]}\n"
            elif ((row[column] < 0 and column in negative_features) or (row[column] > 0 and column in positive_features)) and player_color == "black":
                prompt += f"\t{column}: {row[column]}\n"
    if prompt.endswith("negative changes in the features of the board after the move:\n"):
        prompt += "\tNone\n"

    prompt += "\n"

    #! calculate the difference in features between the last board and the best move board

    if quality_flag == "good":
        pass
    elif quality_flag == "decent":        
        last_board.push_san(str(best_move))
        
        best_move_board_features = board_features(last_board)
        last_board.pop()

        #difference in features
        features_diff = best_move_board_features.subtract(last_board_features)
        #drop the columns that are all 0
        features_diff = features_diff.loc[:, (features_diff != 0).any(axis=0)]

        prompt += "best possible move in this situation: " + best_move + "\n"
        prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(best_move[:2])).symbol() + "\n"
        prompt += "positive changes in the features of the board if the best possible move was made:\n"
        for index, row in features_diff.iterrows():
            for column in features_diff.columns:
                if ((row[column] > 0 and column in positive_features) or (row[column] < 0 and column in negative_features)) and player_color == "white":
                    prompt += f"\t{column}: {row[column]}\n" 
                elif ((row[column] > 0 and column in negative_features) or (row[column] < 0 and column in positive_features)) and player_color == "black":
                    prompt += f"\t{column}: {row[column]}\n"
        if prompt.endswith("positive changes in the features of the board if the best possible move was made:\n"):
            prompt += "\tNone\n"
        
        prompt += "negative changes in the features of the board if the best possible move was made:\n"
        for index, row in features_diff.iterrows():
            for column in features_diff.columns:
                if ((row[column] < 0 and column in positive_features) or (row[column] > 0 and column in negative_features)) and player_color == "white":
                    prompt += f"\t{column}: {row[column]}\n"
                elif ((row[column] < 0 and column in negative_features) or (row[column] > 0 and column in positive_features)) and player_color == "black":
                    prompt += f"\t{column}: {row[column]}\n"
        if prompt.endswith("negative changes in the features of the board if the best possible move was made:\n"):
            prompt += "\tNone\n"
    
    elif quality_flag == "bad":
        current_board.push_san(str(best_counter_move))
        
        best_counter_board_features = board_features(current_board)
        current_board.pop()

        #difference in features
        features_diff = best_counter_board_features.subtract(current_board_features)
        #drop the columns that are all 0
        features_diff = features_diff.loc[:, (features_diff != 0).any(axis=0)]

        prompt += "best possible counter move the opponent could have made after student's move: " + best_counter_move + "\n"
        prompt += "piece moved: " + last_board.piece_at(chess.SQUARE_NAMES.index(best_counter_move[:2])).symbol() + "\n"
        prompt += "positive changes in the features of the board if the best counter opponent move was made:\n"
        for index, row in features_diff.iterrows():
            for column in features_diff.columns:
                if ((row[column] > 0 and column in positive_features) or (row[column] < 0 and column in negative_features)) and player_color == "white":
                    prompt += f"\t{column}: {row[column]}\n" 
                elif ((row[column] > 0 and column in negative_features) or (row[column] < 0 and column in positive_features)) and player_color == "black":
                    prompt += f"\t{column}: {row[column]}\n"
        if prompt.endswith("positive changes in the features of the board if the best counter opponent move was made:\n"):
            prompt += "\tNone\n"
        
        prompt += "negative changes in the features of the board if the best counter opponent move was made:\n"
        for index, row in features_diff.iterrows():
            for column in features_diff.columns:
                if ((row[column] < 0 and column in positive_features) or (row[column] > 0 and column in negative_features)) and player_color == "white":
                    prompt += f"\t{column}: {row[column]}\n"
                elif ((row[column] < 0 and column in negative_features) or (row[column] > 0 and column in positive_features)) and player_color == "black":
                    prompt += f"\t{column}: {row[column]}\n"
        if prompt.endswith("negative changes in the features of the board if the best counter opponent move was made:\n"):
            prompt += "\tNone\n"

    print("prompt")
    print("\n")
    print(prompt)
    print("\n\n\n\n")

    received_message = co.chat(
        message=prompt,
        
        preamble=preamble_template,
        temperature=0.3
    )
    print(received_message.text)
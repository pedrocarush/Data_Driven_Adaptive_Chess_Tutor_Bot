import os
import sys
import math
import chess
from extract_features import board_features
from move_commentator import move_comment, opening_comment

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import pandas as pd

#import chrome_driver_test

def process_game(client,game_id:str):
    print(f"Processing game with id {game_id}")

    board = chess.Board()
    game = client.board.stream_game_state(game_id)

    #driver = chrome_driver_test.initiate_chatbot()
    #driver = None

    #Send first prompt to the chatbot to explain the features and prompt structure
    #give_feature_explanation(driver)

    #! boolean to check if the game is still in the opening
    opening_phase = True

    while True:
        
        try:              
            for event in game:

                if event["type"] == "gameFull":
                    if "id" in event["white"] and event["white"]["id"]  == client.account.get()["id"]:
                        print("You are playing as white")
                        player_color = "white"
                    elif "id" in event["black"] and event["black"]["id"] == client.account.get()["id"]:
                        print("You are playing as black")
                        player_color = "black"
                    

                    user_input = input("Press r to use account rating to choose the model, or input a number to choose another rating\n")

                    user_input = user_input.lower().replace(" ", "")

                    if user_input == "r":
                        player_rating = event[player_color]["rating"]
                    elif user_input.isdigit():
                        player_rating = int(user_input)

                    # round down the rating of the player to choose the model
                    player_rating = int(math.floor(player_rating / 100.0)) * 100
                    
                    # choose the model based on the player rating
                    existent_models = os.listdir("../data/6_models")
                    for model in existent_models:
                        if model.startswith(f"Elo_{player_rating}") and model.endswith(".pkl"):
                            chess_model = tf.keras.models.load_model(f"../data/6_models/{model}")
                            print(f"Model for player rating {player_rating} loaded")
                            break
                    else:
                        print("No model found for player rating, using default model for 1500-1600")
                        chess_model = tf.keras.models.load_model(f"../data/6_models/Elo_1500-1600.csv.pkl")
                        break
                    #choose the opening dataframe based on the player rating
                    existent_openings = os.listdir("../data/opening_stats")
                    for opening in existent_openings:
                        if opening.startswith(f"Elo_{player_rating}") and opening.endswith(".csv"):
                            opening_df = pd.read_csv(f"../data/opening_stats/{opening}")
                            print(f"Opening dataframe for player rating {player_rating} loaded")
                            break
                    else:
                        print("No opening dataframe found for player rating, using default opening dataframe for 1500-1600")
                        opening_df = pd.read_csv(f"../data/opening_stats/Elo_1500-1600.csv")
                        break

                
                elif event["type"] == "gameState":

                    moves = event["moves"].split()
                    
                    if board.fen() == chess.STARTING_FEN:
                        board = chess.Board()
                        for move in moves:
                            board.push_san(move)
                        played_move = moves[-1]
                    else:
                        # find the last move played
                        played_move = moves[-1]
                        board.push_san(played_move)
                    
                    #print(board.fen())
                    
                    #!board.turn is True if it's white's turn and False if it's black
                    #!in this case board is already updated with the last move played so the turn is of the next player not the one who played the last move

                    if ((board.turn and player_color == "black") or (not board.turn and player_color == "white")):

                        #!OPENING PHASE CHECK
                        if opening_phase == True:
                            response = client.opening_explorer.get_lichess_games(position=board.fen())
                            if response["opening"] is not None:
                                print(f"Opening: {response['opening']['name']}")
                                last_board = board.copy()
                                last_board.pop()
                                check_best_opening_move(client,last_board,played_move,player_color,response['opening']['name'],opening_df)
                            else:
                                #print("Not in opening")
                                opening_phase = False

                        #! Normal move comentator (starts at the end of the opening phase)
                        if opening_phase == False:
                            last_board = board.copy()
                            last_board.pop()
                            
                            current_board = board.copy()

                            check_best_move(last_board,played_move,current_board,player_color,chess_model)
                
        except KeyboardInterrupt:
            print("Game inspection interrupted")
            break
        except:
            e = sys.exc_info()
            print(f"Error: {e}")
            break

def check_best_opening_move(client,last_board:chess.Board,played_move:str,player_color:str,opening:str,opening_df:pd.DataFrame):

    print("Checking best opening move for player")
    print(f"Played move: {played_move}")
    print(f"Opening: {opening}")

    #!get the opening dataframe for the opening
    if opening in opening_df['Opening'].values:
        opening_winrate = opening_df[opening_df['Opening'] == opening]['Winrate']
    else:
        print("Opening not found in this ELO's data")
        return
    
    opening_winrate = opening_winrate.values[0]
    print(f"Opening winrate: {opening_winrate}")

    best_opening_move = played_move
    best_opening = opening
    best_opening_winrate = opening_winrate

    possible_moves = last_board.legal_moves
    for x in possible_moves:
        print(f"Checking move: {x}")

        move = chess.Move.from_uci(str(x))
        last_board.push(move)

        response = client.opening_explorer.get_lichess_games(position=last_board.fen())
        if response["opening"] is not None:
            new_opening = response['opening']['name']

            if new_opening in opening_df['Opening'].values:
                new_opening_winrate = opening_df[opening_df['Opening'] == new_opening]['Winrate']
                new_opening_winrate = new_opening_winrate.values[0]
                new_opening_total_games = opening_df[opening_df['Opening'] == new_opening]['Total']
                new_opening_total_games = new_opening_total_games.values[0]

                if new_opening_winrate > best_opening_winrate and new_opening_total_games > 20:
                    best_opening_winrate = new_opening_winrate
                    best_opening_move = str(move)
                    best_opening = new_opening
    
        last_board.pop()

    opening_comment(player_color,last_board,played_move,opening,opening_winrate,best_opening_move,best_opening,best_opening_winrate)



def check_best_move(last_board:chess.Board,played_move:str,current_board:chess.Board,player_color:str,chess_model:tf.keras.models.Model):

    print(f"Last board: {last_board.fen()}")
    print(f"Played move: {played_move}")
    print(f"Current board: {current_board.fen()}")
    print("Checking best move for player")

    last_eval = get_model_evaluation(last_board,player_color,chess_model)

    #move_eval = get_model_evaluation(current_board,player_color,chess_model)

    move_eval,best_counter_move = minimax(current_board,1,-10000,10000,False,player_color,chess_model)

    best_eval,best_move = minimax(last_board,2,-10000,10000,True,player_color,chess_model)

    print(f"Last board evaluation: {last_eval}")
    print(f"Move evaluation: {move_eval}")
    print(f"Best move evaluation: {best_eval} , Best move: {best_move}")

    move_comment(player_color,last_board,last_eval,current_board,played_move,move_eval,best_eval,best_move,best_counter_move)


def get_model_evaluation(board:chess.Board,player_color:str,chess_model:tf.keras.models.Model):

    #! checkmate verification
    if board.is_checkmate():
        #print("Checkmate")

        if (board.turn and player_color == "black") or (not board.turn and player_color == "white"):
            #print("Win")
            return 2
        else:
            return -2

    features = board_features(board)
    #features.sort_index(inplace=True,axis=1)

    #!features is a dataframe with the features of the board
    #!chess_model is the model that will be used to evaluate the board
    #!the model will return a value between 0 and 1, 0 being a loss and 1 being a win for the black player, if the player is white the value will be 1 - value
    #!the value will be the probability of the player winning the game

    #print(f"Features: {features.head()}")

    #convert the pandas dataframe to a tensorflow dataset to be used by the model
    features = tfdf.keras.pd_dataframe_to_tf_dataset(features, label="Result")


    #print(f"Model evaluation: {chess_model}")

    #print(chess_model.summary())

    evaluation = chess_model.predict(features,verbose=0)[0][0]

    if player_color == "white":
        evaluation = 1 - evaluation

    return evaluation

# minimax algorithm with alpha beta pruning in fail-soft mode
def minimax(board:chess.Board,depth:int,alpha:int,beta:int,maximizing:bool,player_color,chess_model):
    if depth == 0 or board.is_checkmate():
        return [get_model_evaluation(board,player_color,chess_model)]

    possible_moves = board.legal_moves
    #print(f"Possible moves: {possible_moves}")

    if maximizing:
        value = -10000 
        for x in possible_moves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            if value < minimax(board,depth-1,alpha,beta,not maximizing,player_color,chess_model)[0]:
                value = minimax(board,depth-1,alpha,beta,not maximizing,player_color,chess_model)[0]
                best_move = str(x)
            #value = max(value,minimax(board,depth-1,alpha,beta,not maximizing,player_color,chess_model)[0])
            board.pop()
            alpha = max(alpha,value)
            if beta <= alpha:
                break
        return value,best_move
    else:
        value = 10000
        for x in possible_moves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            if value > minimax(board,depth-1,alpha,beta,not maximizing,player_color,chess_model)[0]:
                value = minimax(board,depth-1,alpha,beta,not maximizing,player_color,chess_model)[0]
                best_move = str(x)
            #value = min(value,minimax(board,depth-1,alpha,beta,not maximizing,player_color,chess_model)[0])
            board.pop()
            beta = min(beta,value)
            if beta <= alpha:
                break
        return value,best_move


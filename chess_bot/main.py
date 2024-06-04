import berserk
import os
import sys

from game_processor import *

def list_games(client):
    # List the games of the currently authenticated user
    if client.games.get_ongoing() == []:
        print("No ongoing games")
    else:
        games = client.games.get_ongoing()
        for game in games:
            variant = game["variant"]["name"]
            rated = game["rated"]
            opponent = game["opponent"]["username"]
            print(f"{games.index(game)} - {variant}, rated: {rated}, VS {opponent}\n")
    
    # wait for user input (press r to refresh, q to quit)
    
    user_input = input("Press r to refresh, q to quit, numbers to choose game \n")
    if user_input == "q":
        sys.exit(0)
    elif user_input == "r":
        list_games(client)
    elif user_input.isdigit() and int(user_input) < len(games) and int(user_input) >= 0:
        game_index = int(user_input)

        game = games[game_index]

        variant = game["variant"]["name"]
        rated = game["rated"]
        opponent = game["opponent"]["username"]
        game_id = game["gameId"]

        print(f"Selected game {game_index} - {variant}, rated: {rated}, VS {opponent} \n")
        process_game(client, game_id)
    else:
        print("Invalid input")

def main():

    if not os.path.exists("lichess_token.txt"):
        print("Please create a file called lichess_token.txt with your lichess API token")
        return

    with open("lichess_token.txt", "r") as f:
        API_TOKEN = f.read().strip()
    
    try:
        session = berserk.TokenSession(API_TOKEN)
    except:
        e = sys.exc_info()
        print(f"cannot create session: {e}")
        sys.exit(-1)
    
    try:
        client = berserk.Client(session=session)
    except:
        e = sys.exc_info()
        print(f"cannot create lichess client: {e}")
        sys.exit(-1)

    print("Session created")
    list_games(client)


if __name__ == "__main__":
    main()
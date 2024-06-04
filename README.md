# Data_Driven_Adaptive_Chess_Tutor_Bot

This is my thesis work, which I made public for everyone to use, either to try for themselves or to improve if they so choose.

The goal of this project is to create a chess tutor bot that can adapt to the user's skill level and provide feedback on their games.


## How to use the project:
1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt` (you may prefer to use a virtual environment) for both requirements.txt files in the chess_bot and process_data_scripts folders
3. Create a new folder called 'data' in the root directory of the project

### If you want to train the models yourself (only if you have a machine with lots of RAM and lots of time to spare):

4. Create a new folder inside the 'data' folder called '0_lichess_data' and download a dataset from https://database.lichess.org/ and extract it into this folder
5. Run the scripts in the process_data_scripts folder in the following order:
    1. `python3 1_filter_pgn_optimized.py`
    2. `python3 2_pgn_to_csv_optimized.py`
    3. `python3 3_csv_parser.py
    4. `python3 4_process_moves.py`
    5. `python3 5_extract_features.py`
    6. `python3 6_model_generation.py`
(The last 2 scripts take much longer than the others)
After this step you should have a folder called '6_models' in the data folder with the models you trained (You can eliminate the other folders inside data if you want to save space)

### If you want to use the pre-trained models:

I'll try to provide a link with models in the future.

6. Create a Lichess account and get an API token with Board API access, and put that token in a file called 'lichess_token.txt' in the chess_bot directory of the project
7. Create a Cohere account and get the API key (This LLM permits free use for small frequency of requests which works for this project), and put that key in a file called 'cohere_token.txt' in the chess_bot directory of the project
8. Go to the lichess website with your account and create a game against the computer
9. Run the script `python3 main.py` in the chess_bot directory of the project and choose the game you want to connect to
10. Choose the Elo rating you want to use to select the model
11. Play the game while the comments appear after each move in the terminal.



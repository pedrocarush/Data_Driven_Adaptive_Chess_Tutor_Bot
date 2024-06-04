import re
from tqdm import tqdm
import subprocess
import os
import shutil

'''
This script filters the PGN files from Lichess to only keep the games that match the following criteria:
- The game is rated
- The game is either a Rapid, Blitz or Classical game
- The players have an ELO between 100 and 200, 200 and 300, ..., 2900 and 3000
'''

def get_number_of_lines(file_path: str) -> int:

    '''
    This function returns the number of lines of a file
    Parameters:
        file_path (str): The path to the file
    Returns:
        line_count (int): The number of lines of the file
    '''
    print(f"Checking number of lines of file {file_path}")
    command = ["wc", "-l", file_path]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip()
    line_count = int(output.split()[0])
    print(f"The file '{file_path}' has {line_count} lines.")
    return line_count

def process_game(game_text):

    '''
    This function checks if a game matches the criteria and if so, it adds it to the list of filtered games
    Parameters:
        game_text (str): The text of the game
    Returns:
        None
    '''

    global filtered_games
    event_pattern = re.compile(r'\[Event "(.*)"\]')
    event_match = event_pattern.search(game_text)
    if event_match:
        try:
            event = event_match.group(1)
            elo_pattern = re.compile(r'\[(White|Black)Elo "(.*)"\]')
            white_elo_match = elo_pattern.search(game_text, 0)
            black_elo_match = elo_pattern.search(game_text, white_elo_match.end())
            # if (event == "Rated Classical game" or event == "Rated Rapid game") and \
            # if (event == "Rated Rapid game") and \
            # if (event == "Rated Classical game") and \
            if (event == "Rated Classical game" or event == "Rated Rapid game" or event == 'Rated Blitz game') and \
            white_elo_match and black_elo_match and \
            MIN_ELO <= int(white_elo_match.group(2)) <= MAX_ELO and MIN_ELO <= int(black_elo_match.group(2)) <= MAX_ELO:
                filtered_games.append(game_text)
                # return game_text

        except:
            global SKIPPING_GLOBAL
            SKIPPING_GLOBAL += 1


def main(num_lines,pgn_file):

    '''
    This function reads the PGN file in chunks and calls the function to process each game
    Parameters:
        num_lines = number of lines from file
        pgn_file = path to pgn file
    Returns:
        None
    '''

    #num_lines = get_number_of_lines(pgn_file)
    chunk_size = int(102400000 / 3)
    total_games = 0  
    global filtered_games
    filtered_games = []

    with open(pgn_file) as f:
        print("Opening file to process")
        game_lines = []
        output_lines_number = 0
        for chunk_start in tqdm(range(0, num_lines, chunk_size)):
            # print(f"Have found {len(filtered_games)} that match the criteria, so far")
            chunk_end = min(chunk_start + chunk_size, num_lines)
            f.seek(chunk_start)

            for _ in range(chunk_end - chunk_start):
                line = f.readline()
                if line.startswith("[Event "):
                    # process the previous game
                    if game_lines:
                        game_text = "".join(game_lines)
                        process_game(game_text)
                        output_lines_number += 1
                    game_lines = [line]
                else:
                    game_lines.append(line)

            # Write the filtered games to a new PGN file
            #print("Writing chunk to output file")
            with open(OUTPUT_FILE, "a") as f_out:
                f_out.write("\n\n".join(filtered_games))
            total_games += len(filtered_games)
            filtered_games = []
            
        # process the last game
        if game_lines:
            game_text = "".join(game_lines)
            process_game(game_text)
            output_lines_number += 1
            
    # Write the last game to the PGN filtered file
    #print("Writing last game to output file")
    with open(OUTPUT_FILE, "a") as f_out:
        f_out.write("\n\n".join(filtered_games))
    total_games += len(filtered_games)

    print(f"Output file has {total_games} games that matched the criteria.")
    print(f"Skipped {SKIPPING_GLOBAL}")



if __name__ =='__main__':

    '''
        Here we define the parameters of the script
        
        MIN_ELO: The minimum ELO of the players
        MAX_ELO: The maximum ELO of the players
        BASE_FOLDER: The folder where the PGN files are located
        OUTPUT_FOLDER: The folder where the filtered PGN files will be saved

        The script will iterate over all the PGN files in the BASE_FOLDER and will create a new PGN file for each ELO range in the OUTPUT_FOLDER

        For example, if MIN_ELO = 100 and MAX_ELO = 200, the script will create a new PGN file in the OUTPUT_FOLDER with the games where both players have an ELO between 100 and 200
    '''
    BASE_FOLDER = '../data/0_lichess_data'
    OUTPUT_FOLDER = '../data/1_lichess_data_filtered'

    #Delete the output folder to garantee that it is empty, otherwise the script will append to the existing file which could lead to repeated games
    try:
        shutil.rmtree('../data/1_lichess_data_filtered')
    except Exception as e:
        print(f'Failed to delete directory: {e}')

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for pgn_file_name in tqdm(os.listdir(BASE_FOLDER)):
        MIN_ELO = 100
        MAX_ELO = 200
        print(pgn_file_name)
        pgn_file = f'{BASE_FOLDER}/{pgn_file_name}'
        num_lines = get_number_of_lines(pgn_file)
        while MAX_ELO <= 3000:
            OUTPUT_FILE = f'{OUTPUT_FOLDER}/Elo_{MIN_ELO}_{MAX_ELO}.pgn'
            print(F'Min/max elo being extracted: {MIN_ELO}/{MAX_ELO} ')
            SKIPPING_GLOBAL = 0
            
            main(num_lines,pgn_file)
            MIN_ELO += 100
            MAX_ELO += 100
    
   
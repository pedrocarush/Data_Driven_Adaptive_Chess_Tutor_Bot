import cohere

f = open("cohere_key.txt","r")

api_key = f.readline()

co = cohere.Client(api_key)
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

preamble_template = '''

## Task & Context
You are a chess tutor giving advice to a student. 
You will be making comments on the moves made by the student to help them improve their chess skills. 
The comments will be based on the changes in the features of the board after the move. Don't comment on anything that isn't present in the prompt given.
You will also be receive information about the best possible move in that situation and the changes in the features of the board if that move was made, to help the student understand the best possible move in that situation. Always specify the piece moved and the square it moved to.
And you will also be receiving information about the best possible counter move for the opponent after the student's move, to help the student understand the consequences of their move. Always specify the piece moved and the square it moved to.

## Style Guide
Try to keep the comments concise and to the point, while also keeping positive and encouraging.

## Meaning of each Feature

'''
preamble_template += '\n'.join([f'- **{k}**: {v}' for k, v in feature_explanation.items()])

good_prompt = '''

## Instructions
You are a chess tutor giving advice to a student. Make a comment on the move made by the student considering the following:

## Input
student is playing white side.
quality of the move made by the student : good
positive changes in the features of the board after the move:
  n_black:-1
	Trapped_rooks_white:-1
	King_zone_attacked_squares_white:-1
	Mobility_black:-3
	Mobility_white:+2
negative changes in the features of the board after the move:
  King_pawn_shield_white:-1
  King_zone_controlled_squares_white:-1
'''

decent_prompt = '''

## Instructions
You are a chess tutor giving advice to a student. Make a comment on the move made by the student and comment on the best possible move in that situation as well, considering the following:

## Input
student is playing white side.
quality of the move made by the student : decent
positive changes in the features of the board after the move:
  Mobility_black:-2
  Mobility_white:+1
negative changes in the features of the board after the move:
  King_zone_controlled_squares_white:-1
  King_zone_attacked_squares_white:-1

best possible move in this situation: c3d4
piece moved: knight
positive changes in the features of the board if the best possible move was made:
  n_black:-1
  Trapped_rooks_white:-1
  Mobility_black:-3
  Mobility_white:+2
negative changes in the features of the board if the best possible move was made:
  King_pawn_shield_white:-1
  King_zone_controlled_squares_white:-1
'''

bad_prompt = '''

## Instructions
You are a chess tutor giving advice to a student. Make a comment on the move made by the student and comment on the best possible counter move for the opponent after the student's move, considering the following:

## Input
student is playing white side.
quality of the move made by the student : bad
positive changes in the features of the board after the move:
  Mobility_black:-2
  Mobility_white:+1
negative changes in the features of the board after the move:
  King_zone_controlled_squares_white:-1
  King_zone_attacked_squares_white:-1

best possible counter move the opponent could have made after student's move: c4d5
piece moved: p
positive changes in the features of the board if the best counter opponent move was made:
  Mobility_black:-2
  Mobility_white:+1
negative changes in the features of the board if the best counter opponent move was made:
  q_white:-1
  King_zone_controlled_squares_black:+1
  King_zone_attacked_squares_white:+1
'''


response = co.chat(
  message=bad_prompt,
  
  preamble=preamble_template,
  temperature=0.3
)

print(response.text)
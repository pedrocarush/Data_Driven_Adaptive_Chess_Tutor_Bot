import pandas as pd
import chess
import numpy as np

'''
    Collection of functions to extract features from the board
    Used by chess_bot/extract_features.py, process_data_scripts/5_extract_features.py

    Code was reused and readapted from https://github.com/scchess/evAl-chess

    The following functions are included:
    - attack_and_defend_maps
    - sliding_pieces_mobility
    - piece_lists
    - side_to_move
    - castling_rights


    Currently, the following features are extracted:
    - count_pieces
    - mobility
    - pawn_structure
    - king_safety
    - trapped_pieces
    - check

'''

# Add some crucial constants to the `chess` module.
chess.WHITE_PIECES, chess.BLACK_PIECES = (
    ('P', 'N', 'B', 'R', 'Q', 'K'),
    ('p', 'n', 'b' ,'r', 'q', 'k')
)
chess.PIECES = chess.WHITE_PIECES + chess.BLACK_PIECES
chess.SLIDING_PIECES = (
    'B', 'R', 'Q', 'b', 'r', 'q'
)
#Max number of pieces of each type (extra knight and queen)
#!TODO: Check if this changes values
chess.PIECE_CAPACITY = {
    'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
    'p' : 8, 'n' : 3, 'b' : 2, 'r' : 2, 'q' : 2, 'k' : 1
}
chess.MISSING_PIECE_SQUARE = -1
chess.PIECE_MOVEMENTS = {
    'R' : tuple(zip((+1, +0, -1, +0), (+0, +1, +0, -1))),
    'B' : tuple(zip((+1, -1, -1, +1), (+1, +1, -1, -1))),
    'N' : tuple(
        zip(
            (+2, +1, -1, -2, -2, -1, +1, +2),
            (+1, +2, +2, +1, -1, -2, -2, -1)
        )
    ),
    'P' : tuple(zip((-1, -1), (-1, +1))),
    'p' : tuple(zip((+1, +1), (-1, +1)))
}
chess.PIECE_MOVEMENTS['Q'] = chess.PIECE_MOVEMENTS['K'] = (
    chess.PIECE_MOVEMENTS['B'] + chess.PIECE_MOVEMENTS['R']
)
for piece in ('k', 'q', 'r', 'b', 'n'):
    chess.PIECE_MOVEMENTS[piece] = chess.PIECE_MOVEMENTS[piece.upper()]

def __init_attackers_and_scope(board, piece_squares):
    '''
    Manually calculates the value of the lowest-valued attacker and
    defender of each square and the scope of each sliding piece in
    `board`. Stores this information as:

        `board.min_attacker_of`:
            The value of the lowest-valued attacker of each square for
            each color. `board.min_attacker_of[square][chess.BLACK]`
            is the value of the lowest-valued black piece that attacks
            `square`.

        `board.sliding_piece_scopes`:
            How far each sliding piece can slide in each direction
            before either hitting a piece or the edge of the board.
            If it hits a piece of the opposite color, the square
            the piece is on counts as a square onto which it can slide.

    '''
    # The color of each piece on each square -- -1 if the square is
    # empty.
    piece_colors = np.full(shape=(8, 8), fill_value=-1, dtype=int)
    for piece in piece_squares:
        for square in piece_squares[piece]:
            piece_colors[__to_coord(square)] = (
                chess.WHITE if piece in chess.WHITE_PIECES
                else chess.BLACK
            )

    # The value of the lowest-valued attacker of each square.
    min_white_attacker_of, min_black_attacker_of = (
        np.zeros((8, 8)),
        np.zeros((8, 8))
    )

    def in_range(i, j):
        '''
        Whether the row-major coordinate `(i, j)` exists.
        '''
        return (0 <= i < 8) and (0 <= j < 8)

    def assign(arr, i, j, val):
        '''
        Returns:
            3-d tuple
                The first element is a bool that is `True` if the
                square exists; the second is a bool that is `True`
                if the square has a piece on it; the third is the color
                of the piece if the second is `True` and `None`
                otherwise.
        '''
        if not in_range(i, j):
            return False, False, None
        elif piece_colors[i, j] != -1:
            arr[i, j] = val
            return True, True, piece_colors[i, j]
        else:
            arr[i, j] = val
            return True, False, None

    def assign_while(arr, piece_color, i, di, j, dj, val):
        '''
        Simulates a sliding piece moving.

        Starts at `(i, j)` and iterates `(di, dj)` until it hits a
        piece or the edge of the board. Assigns its value to each
        square it visited.

        Returns:
            `int`
                The number of times it assigned a square -- the scope
                of the piece at `(i, j)`.
        '''
        continue_assigning, scope = True, 0
        while continue_assigning:
            exists, had_piece, other_piece_color = assign(
                arr, i + di, j + dj, val
            )
            continue_assigning = exists and not had_piece
            scope += (
                (exists and not had_piece)
                or (had_piece and not other_piece_color == piece_color)
            )
            i += di
            j += dj
        return scope

    # The relative value of each piece.
    relative_vals = {
        'P' : 1, 'N' : 2, 'B' : 3, 'R' : 4, 'Q' : 5, 'K' : 6,
        'p' : 1, 'n' : 2, 'b' : 3, 'r' : 4, 'q' : 5, 'k' : 6
    }

    # How far each sliding piece can move in each direction.
    board.sliding_piece_scopes = {
        (sliding_piece, square) : []
        for sliding_piece in chess.SLIDING_PIECES
        for square in piece_squares[sliding_piece]
    }

    # Iterate through all legal moves of each piece, beginning with the
    # highest value, assigning the piece's value to the `attackers`
    # arrays. The result is the value of the lowest-valued attacker
    # for each square.
    for piece in reversed(chess.PIECES):
        piece_color = (
            chess.WHITE
            if piece in chess.WHITE_PIECES
            else chess.BLACK
        )
        # Which array to which to assign.
        arr = (
            min_white_attacker_of
            if piece_color == chess.WHITE
            else min_black_attacker_of
        )
        # If it's a sliding piece, assign its value in each direction
        # while it can continue moving in the direction.
        if piece in chess.SLIDING_PIECES:
            for square in piece_squares[piece]:
                i, j = __to_coord(square)
                for di, dj in chess.PIECE_MOVEMENTS[piece]:
                    scope = assign_while(
                        arr, piece_color,
                        i, di,
                        j, dj,
                        relative_vals[piece]
                    )
                    board.sliding_piece_scopes[(piece, square)].append(
                        scope
                    )
        # If it's not a sliding piece, simply iterate through each of
        # its movements and assign its value.
        else:
            for i, j in (
                __to_coord(square)
                for square in piece_squares[piece]
            ):
                for di, dj in chess.PIECE_MOVEMENTS[piece]:
                    assign(arr, i + di, j + dj, relative_vals[piece])

    board.min_attacker_of = [
        (j, i)
        for i, j in zip(
            min_white_attacker_of.flatten().astype(int).tolist(),
            min_black_attacker_of.flatten().astype(int).tolist()
        )
    ]

def _init_square_data(board):
    '''
    Calculates some basic information of the board and stores it in
    `board`.

    `board.piece_squares`:

        Each possible piece -- 8 pawns, 3 knights, 2 queens, etc. --
        and its square. If the piece isn't on the board, the square is
        set to `chess.MISSING_PIECE_SQUARE`. The length is constant
        regardless of the board because the number of possible
        pieces is constant.

        Pieces are grouped together in the same order as `chess.PIECES`
        -- 'P', 'N', 'B' ... 'p', 'n', 'b' ... -- but their squares are
        randomly permuted. As a result, the first 8 pieces are
        guaranteed to be 'P' but their squares random.
    '''
    # The squares of the pieces on the board.
    piece_squares = { piece : [] for piece in chess.PIECES }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_squares[piece.symbol()].append(square)

    # Pass `piece_squares` before adding the missing pieces' squares.
    # This is a bit ugly, I know.
    __init_attackers_and_scope(board, piece_squares)

    # Add the missing pieces and their squares,
    # `chess.MISSING_PIECE_SQUARE`.
    for piece in chess.PIECES:
        piece_squares[piece] += (
            [chess.MISSING_PIECE_SQUARE]
            * (chess.PIECE_CAPACITY[piece] - len(piece_squares[piece]))
        )

    # Set to `board.piece_squares` with the pieces ordered correctly
    # and the squares of each piece permuted.
    board.piece_squares = [
        (piece, square)
        for piece in chess.PIECES
        for square in np.random.permutation(piece_squares[piece]).tolist()
    ]

def __to_coord(square):
    '''
    The row-major coordinate of square. Used by `_piece_lists()`.
    Example: `__to_coord(chess.A1) == (0, 0)`.
            `__to_coord(chess.H8) == (7, 7)`.
    '''
    return (square // 8, square % 8)

def __direction(from_square, to_square):
    '''
    The direction traveled in going from `from_square` to `to_square`.
    The value v returned yields the direction in <cos(v * pi / 4),
    sin(v * pi / 4)>.
    '''
    from_coord, to_coord = __to_coord(from_square), __to_coord(to_square)
    dx, dy = (to_coord[0] - from_coord[0], to_coord[1] - from_coord[1])
    dx, dy = dx // max(abs(dx), abs(dy)), dy // max(abs(dx), abs(dy))
    return {
        (0, 1) : 0,
        (-1, 1) : 1,
        (-1, 0) : 2,
        (-1, -1) : 3,
        (0, -1) : 4,
        (1, -1) : 5,
        (1, 0) : 6,
        (1, 1) : 7
    }[(dx, dy)]

def attack_and_defend_maps(board):
    '''
    The value of the lowest-valued attack and defender of each square;
    by default, 0.
    These maps are in the perspective of white.
    so the attack map is the value of the lowest-valued black piece attacking the square.
    and the defend map is the value of the lowest-valued white piece defending the square.

    For the following 4x4 board, R B . .
                                 . . b .
                                 q k . .
                                 . . . P,
    the attack map would be  5 3 5 3
                             5 5 6 0
                             6 3 6 3
                             3 5 6 0,
    and the defend map  4 4 0 0
                        3 0 3 0
                        4 0 1 3
                        0 0 0 0.

    Number of features contributed: 128. Sixty-four integers for each
    the attack and defend maps.
    '''

    attack_and_defend_maps_dict = dict()
    """
    example:
    attack_and_defend_maps_dict = {
        0_0_min_attacker: 5,
        0_0_min_defender: 4,
        1_0_min_attacker: 3,
        1_0_min_defender: 4,
        ...
    }
    """

    for square in chess.SQUARES:
        attack_and_defend_maps_dict[f'{square//8}_{square%8}_min_attacker'] = np.int8(board.min_attacker_of[square][chess.BLACK])
        attack_and_defend_maps_dict[f'{square//8}_{square%8}_min_defender'] = np.int8(board.min_attacker_of[square][chess.WHITE])


    #attack_and_defend_maps = [
    #    board.min_attacker_of[square][color]
    #    for color in (chess.BLACK, chess.WHITE)
    #    for square in chess.SQUARES
    #]

    return attack_and_defend_maps_dict

def sliding_pieces_mobility(board):
    '''
    How far each {white, black} {bishop, rook, queen} can slide in each
    legal direction.

    Number of features contributed: One `int` for each direction
    for each possible piece. 8 directions for each existing sliding piece.
    If a piece doesn't exist or is not a sliding piece, the value is -1 for all 8 directions.
    '''

    mobilities_dict = dict()
    """
    example:
    queen
    mobilities_dict = {
        0_white_D1: 5,
        0_white_D2: 2,
        0_white_D3: 3,
        0_white_D4: 1,
        0_white_D5: 0,
        0_white_D6: 1,
        0_white_D7: 1,
        0_white_D8: 2,
        0_black_D1: 3,
        ...
    }
    """

    current_piece_white = 0
    current_piece_black = 0
    for piece,square in board.piece_squares:
        if square != chess.MISSING_PIECE_SQUARE:
            number_of_directions = 0
            if piece in chess.SLIDING_PIECES:
                for scope_dir in board.sliding_piece_scopes[(piece, square)]:
                    mobilities_dict[f'{current_piece_white}_white_D{number_of_directions}' if board.piece_at(square).color == chess.WHITE else f'{current_piece_black}_black_D{number_of_directions}'  ] = np.int8(scope_dir)
                    number_of_directions += 1
                while number_of_directions < 8:
                    mobilities_dict[f'{current_piece_white}_white_D{number_of_directions}' if board.piece_at(square).color == chess.WHITE else f'{current_piece_black}_black_D{number_of_directions}'  ] = np.int8(-1)
                    number_of_directions += 1
        
            if board.piece_at(square).color == chess.WHITE:
                while number_of_directions < 8:
                    mobilities_dict[f'{current_piece_white}_white_D{number_of_directions}'] = np.int8(-1)
                    number_of_directions += 1
                current_piece_white += 1
            else:
                while number_of_directions < 8:
                    mobilities_dict[f'{current_piece_black}_black_D{number_of_directions}'] = np.int8(-1)
                    number_of_directions += 1
                current_piece_black += 1

    while current_piece_white < 16:
        for number_of_directions in range(8):
            mobilities_dict[f'{current_piece_white}_white_D{number_of_directions}'] = np.int8(-1)
        current_piece_white += 1
    
    while current_piece_black < 16:
        for number_of_directions in range(8):
            mobilities_dict[f'{current_piece_black}_black_D{number_of_directions}'] = np.int8(-1)
        current_piece_black += 1

    #mobilities = [
    #    scope_dir
    #    for piece, square in board.piece_squares
    #    if piece in chess.SLIDING_PIECES
    #    for scope_dir in (
    #        board.sliding_piece_scopes[(piece, square)]
    #        if square != chess.MISSING_PIECE_SQUARE
    #        else [-1] * len(chess.PIECE_MOVEMENTS[piece])
    #    )
    #]

    return mobilities_dict

def piece_lists(board):
    '''
    For each possible piece (*):
    1. Its type. By default, -1. (This is a dummy value.) (Can be P, N, B, R, Q, K, p, n, b, r, q, k)
    2. Its row-major, zero-indexed coordinate. By default, (-1, -1).
    3. Whether the piece is on the board.
    4. The values of the minimum-valued {attacker, defender} of the
       piece stored in a tuple. By default, (-1, -1).

    Number of features contributed: 16 * (1 + 2 + 1 + 2) * 2 = 192
    6 features for each one of the 16 pieces of each color.
    '''

    piece_lists_dict = dict()
    """
    example:
    pawn
    piece_lists = {
        0_white_type : 'P',
        0_white_row : 4,
        0_white_col : 2,
        0_white : 1,
        0_white_min_attacker : 0,
        0_white_min_defender : 2, 
    
    """
    current_piece_white = 0
    current_piece_black = 0
    for piece, square in board.piece_squares:
        if square != chess.MISSING_PIECE_SQUARE:
            if board.piece_at(square).color == chess.WHITE:
                feature_key = str(current_piece_white) + '_white' 
                piece_lists_dict[feature_key + '_min_attacker'], piece_lists_dict[feature_key + '_min_defender'] = np.int8(board.min_attacker_of[square])
                #update current piece because the current is already in feature_key
                current_piece_white += 1
            else:
                feature_key = str(current_piece_black) + '_black'
                piece_lists_dict[feature_key + '_min_attacker'], piece_lists_dict[feature_key + '_min_defender'] = np.int8(tuple(reversed(board.min_attacker_of[square])))
                #update current piece because the current is already in feature_key
                current_piece_black += 1

            piece_lists_dict[feature_key + '_type'] = piece
            piece_lists_dict[feature_key + '_row'], piece_lists_dict[feature_key + '_col'] = np.int8(__to_coord(square))
            piece_lists_dict[feature_key] = 1

    while current_piece_black < 16:
        feature_key = str(current_piece_black) + '_black'
        piece_lists_dict[feature_key + '_type'] = "X"
        piece_lists_dict[feature_key + '_row'], piece_lists_dict[feature_key + '_col'] = np.int8(-1), np.int8(-1)
        piece_lists_dict[feature_key] = 0
        piece_lists_dict[feature_key + '_min_attacker'], piece_lists_dict[feature_key + '_min_defender'] = np.int8(-1), np.int8(-1)
        current_piece_black += 1
    
    while current_piece_white < 16:
        feature_key = str(current_piece_white) + '_white'
        piece_lists_dict[feature_key + '_type'] = "X"
        piece_lists_dict[feature_key + '_row'], piece_lists_dict[feature_key + '_col'] = np.int8(-1), np.int8(-1)
        piece_lists_dict[feature_key] = 0
        piece_lists_dict[feature_key + '_min_attacker'], piece_lists_dict[feature_key + '_min_defender'] = np.int8(-1), np.int8(-1)
        current_piece_white += 1


    #piece_lists = list(
    #    sum(
    #        [
    #            (-1, -1, False, -1, -1)
    #            if square == chess.MISSING_PIECE_SQUARE
    #            else (
    #                __to_coord(square)
    #                + (True, )
    #                + (
    #                    board.min_attacker_of[square]
    #                    if board.piece_at(square).color == chess.WHITE
    #                    else tuple(reversed(board.min_attacker_of[square]))
    #                )
    #            )
    #            for piece, square in board.piece_squares
    #        ],
    #        tuple()
    #    )
    #)

    return piece_lists_dict

def side_to_move(board: chess.Board):
    '''
    True if it's White turn to move.

    Number of features contributed: 1.
    '''
    return {"SideToMove":np.int8(board.turn)}

def castling_rights(board:chess.Board):
    '''
    Castling rights for both players.
    True if {White,BLack} the player can castle in {kingside,queenside}.

    Number of features contributed: 4.
    '''
    
    return {"white_kingside_castling_rights":np.int8(board.has_kingside_castling_rights(chess.WHITE)),
                          "white_queenside_castling_rights":np.int8(board.has_queenside_castling_rights(chess.WHITE)),
                          "black_kingside_castling_rights":np.int8(board.has_kingside_castling_rights(chess.BLACK)),
                          "black_queenside_castling_rights":np.int8(board.has_queenside_castling_rights(chess.BLACK))
                          }

def count_pieces(fen: str) -> dict:
    '''
    The number of each piece on the board.

    Number of features contributed: 12. Six types of pieces for each
    side.
    '''

    piece_count = {'K_white': 0, 'Q_white': 0, 'R_white': 0, 'B_white': 0, 'N_white': 0, 'P_white': 0, 'k_black': 0, 'q_black': 0, 'r_black': 0, 'b_black': 0, 'n_black': 0, 'p_black': 0}

    fen_parts = fen.split()
    board_state = fen_parts[0]
    rows = board_state.split('/')
    
    # Define a lookup table to convert a piece character to its corresponding count key
    piece_lookup = {
        'K': 'K_white',
        'Q': 'Q_white',
        'R': 'R_white',
        'B': 'B_white',
        'N': 'N_white',
        'P': 'P_white',
        'k': 'k_black',
        'q': 'q_black',
        'r': 'r_black',
        'b': 'b_black',
        'n': 'n_black',
        'p': 'p_black'
    }
    
    for i, row in enumerate(rows):
        rank = 8 - i
        file = 0
        for chr in row:
            if chr.isnumeric():
                file += int(chr)
            else:
                piece_count[piece_lookup[chr]] += 1
                file += 1

    for piece in piece_count:
        piece_count[piece] = np.int8(piece_count[piece])

    return piece_count

def mobility(board):

    # check turn
    if board.turn == chess.WHITE:
    
        number_of_moves_white = len(list(board.legal_moves))
        board.turn = chess.BLACK
        number_of_moves_black = len(list(board.legal_moves))
        board.turn = chess.WHITE

    else:
        
        number_of_moves_black = len(list(board.legal_moves))
        board.turn = chess.WHITE
        number_of_moves_white = len(list(board.legal_moves))
        board.turn = chess.BLACK

    return {"Mobility_white":np.int8(number_of_moves_white), "Mobility_black":np.int8(number_of_moves_black)}

def pawn_structure(board):
    # list of ideas
    # passed pawns (which pawns are free to move forward without being blocked by an enemy pawn) - positive
    # isolated pawns (pawns that don't have a friendly pawn on the adjacent files) - negative
    # doubled pawns (pawns that are on the same file) - negative

    passed_pawns_white = 0
    isolated_pawns_white = 0
    doubled_pawns_white = 0
    passed_pawns_black = 0
    isolated_pawns_black = 0
    doubled_pawns_black = 0
    
    for piece, square in board.piece_squares:
        if square != chess.MISSING_PIECE_SQUARE:
            if piece == 'P' or piece == 'p':

                #! PASSED PAWNS
                # check if the pawn is passed
                # need to check if the pawn is blocked by an enemy pawn on the adjacent files and in the current file
                # if the pawn is not blocked by an enemy pawn, then it is passed

                coordinate = __to_coord(square)
                min_file = max(0, coordinate[1] - 1)
                max_file = min(7, coordinate[1] + 1)
                if board.piece_at(square).color == chess.WHITE:
                    min_rank = coordinate[0]
                    max_rank = 7
                else:
                    min_rank = 0
                    max_rank = coordinate[0]

                for f in range(min_file, max_file + 1):
                    for rank in range(min_rank, max_rank + 1):
                        if board.piece_at(chess.square(f, rank)) == chess.Piece(chess.PAWN, not board.piece_at(square).color):
                            break
                    else:
                        continue
                    break
                else:
                    if board.piece_at(square).color == chess.WHITE:
                        passed_pawns_white += 1
                    else:
                        passed_pawns_black += 1
                
                #! ISOLATED PAWNS
                    
                # check if the pawn is isolated
                # need to check if the pawn has a friendly pawn on the adjacent files
                # if the pawn doesn't have a friendly pawn on the adjacent files, then it is isolated
                
                min_file = max(0, coordinate[1] - 1)
                max_file = min(7, coordinate[1] + 1)
                min_rank = max(0, coordinate[0] - 1)
                max_rank = min(7, coordinate[0] + 1)
                isolated = True

                for f in range(min_file, max_file + 1):
                    for rank in range(min_rank, max_rank + 1):
                        if board.piece_at(chess.square(f, rank)) == chess.Piece(chess.PAWN, board.piece_at(square).color) and (rank, f) != coordinate:
                            isolated = False
                            break
                    else:
                        continue
                    break
                if isolated:
                    if board.piece_at(square).color == chess.WHITE:
                        isolated_pawns_white += 1
                    else:
                        isolated_pawns_black += 1
                
                #! DOUBLED PAWNS
                # check if the pawn is doubled
                # need to check if there is another friendly pawn on the same file
                # if there is another friendly pawn on the same file, then it is doubled
                
                if board.piece_at(square).color == chess.WHITE:
                    min_rank = min(7, coordinate[0] + 1)
                    max_rank = 7
                else:
                    max_rank = max(0, coordinate[0] - 1)
                    min_rank = 0

                for rank in range(min_rank, max_rank + 1):
                    if board.piece_at(chess.square(coordinate[1], rank)) == chess.Piece(chess.PAWN, board.piece_at(square).color) and rank != coordinate[0]:
                        if board.piece_at(square).color == chess.WHITE:
                            doubled_pawns_white += 1
                        else:
                            doubled_pawns_black += 1
                        break



    return {"Passed_pawns_white":np.int8(passed_pawns_white), "Isolated_pawns_white":np.int8(isolated_pawns_white), "Doubled_pawns_white":np.int8(doubled_pawns_white), "Passed_pawns_black":np.int8(passed_pawns_black), "Isolated_pawns_black":np.int8(isolated_pawns_black), "Doubled_pawns_black":np.int8(doubled_pawns_black)}

def king_safety(board):
    # list of ideas
    # king pawn shield (number of friendly pawns in front or side of the king) - positive
    # number of squares around the king that are attacked by enemy pieces - negative
    # number of squares around the king that are controlled by friendly pieces - positive

    number_of_kings = 0

    #! King Pawn Shield (number of friendly pawns in front or side of the king) - positive
    king_pawn_shield_white = 0
    king_pawn_shield_black = 0

    #! King Zone Attacked Squares (number of squares around the king that are attacked by enemy pieces) - negative
    king_zone_attacked_squares_white = 0
    king_zone_attacked_squares_black = 0

    #! King Zone Controlled Squares (number of squares around the king that are controlled by friendly pieces) - positive
    king_zone_controlled_squares_white = 0
    king_zone_controlled_squares_black = 0


    for piece, square in board.piece_squares:
        if square != chess.MISSING_PIECE_SQUARE:
            if piece == 'K' or piece == 'k':
                number_of_kings += 1

                coordinate = __to_coord(square)

                #! KING PAWN SHIELD
                min_file = max(0, coordinate[1] - 1)
                max_file = min(7, coordinate[1] + 1)
                if board.piece_at(square).color == chess.WHITE:
                    min_rank = coordinate[0]
                    max_rank = min(7, coordinate[0] + 2)
                else:
                    min_rank = max(0, coordinate[0] - 2)
                    max_rank = coordinate[0]

                for f in range(min_file, max_file + 1):
                    for rank in range(min_rank, max_rank + 1):
                        if board.piece_at(chess.square(f, rank)) == chess.Piece(chess.PAWN, board.piece_at(square).color):
                            if board.piece_at(square).color == chess.WHITE:
                                king_pawn_shield_white += 1
                            else:
                                king_pawn_shield_black += 1

                #! King Zone Attacked Squares
                # check if the squares around the king are attacked by enemy pieces
                #! King Zone Controlled Squares
                # check if the squares around the king are controlled by friendly pieces
                
                max_file = min(7, coordinate[1] + 1)
                min_file = max(0, coordinate[1] - 1)
                max_rank = min(7, coordinate[0] + 1)
                min_rank = max(0, coordinate[0] - 1)

                for f in range(min_file, max_file + 1):
                    for rank in range(min_rank, max_rank + 1):
                        if board.min_attacker_of[chess.square(f, rank)][not board.piece_at(square).color] not in [0,6] :
                            if board.piece_at(square).color == chess.WHITE:
                                king_zone_attacked_squares_white += 1
                            else:
                                king_zone_attacked_squares_black += 1
                        if board.min_attacker_of[chess.square(f, rank)][board.piece_at(square).color] not in [0,6]:
                            if board.piece_at(square).color == chess.WHITE:
                                king_zone_controlled_squares_white += 1
                            else:
                                king_zone_controlled_squares_black += 1


                
            
        if number_of_kings == 2:
            break



    return  {"King_pawn_shield_white":np.int8(king_pawn_shield_white), "King_pawn_shield_black":np.int8(king_pawn_shield_black), "King_zone_attacked_squares_white":np.int8(king_zone_attacked_squares_white), "King_zone_attacked_squares_black":np.int8(king_zone_attacked_squares_black), "King_zone_controlled_squares_white":np.int8(king_zone_controlled_squares_white), "King_zone_controlled_squares_black":np.int8(king_zone_controlled_squares_black)}

def trapped_pieces(board):

    #! Trapped sliding pieces (sliding pieces that have low mobility) - negative

    Trapped_bishops_white = 0
    Trapped_bishops_black = 0

    Trapped_rooks_white = 0
    Trapped_rooks_black = 0

    Trapped_queens_white = 0
    Trapped_queens_black = 0

    for piece, square in board.piece_squares:
        if square != chess.MISSING_PIECE_SQUARE:
            if piece == 'B' or piece == 'b':
                if board.piece_at(square).color == chess.WHITE:
                    if board.sliding_piece_scopes[(piece, square)].count(0) >= 3:
                        Trapped_bishops_white += 1
                else:
                    if board.sliding_piece_scopes[(piece, square)].count(0) >= 3:
                        Trapped_bishops_black += 1
            elif piece == 'R' or piece == 'r':
                if board.piece_at(square).color == chess.WHITE:
                    if board.sliding_piece_scopes[(piece, square)].count(0) >= 3:
                        Trapped_rooks_white += 1
                else:
                    if board.sliding_piece_scopes[(piece, square)].count(0) >= 3:
                        Trapped_rooks_black += 1
            elif piece == 'Q' or piece == 'q':
                if board.piece_at(square).color == chess.WHITE:
                    if board.sliding_piece_scopes[(piece, square)].count(0) >= 7:
                        Trapped_queens_white += 1
                else:
                    if board.sliding_piece_scopes[(piece, square)].count(0) >= 7:
                        Trapped_queens_black += 1



    #! Trapped knights (knights that have low mobility) - negative NOT DONE


    return {"Trapped_bishops_white":np.int8(Trapped_bishops_white), "Trapped_bishops_black":np.int8(Trapped_bishops_black), "Trapped_rooks_white":np.int8(Trapped_rooks_white), "Trapped_rooks_black":np.int8(Trapped_rooks_black), "Trapped_queens_white":np.int8(Trapped_queens_white), "Trapped_queens_black":np.int8(Trapped_queens_black)}

def check(board):
    '''
    Check if the each player is in check
    '''

    if board.turn == chess.WHITE:
        return {"Check_white":np.int8(board.is_check()), "Check_black":np.int8(0)}
    
    else:
        return {"Check_white":np.int8(0),"Check_black":np.int8(board.is_check())}


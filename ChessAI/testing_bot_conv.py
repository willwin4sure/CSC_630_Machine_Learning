import chess
import chess.polyglot
import random
import numpy as np
import torch
import math
from math import log10

def pawn_advantage_to_prob(adv):
    if adv < -1000:
        return 0
    return 1/(1+10**(adv/(-4)))

def prob_to_pawn_advantage(prob):
    if prob < 1e-100:
        return -1000000
    if prob > 1-1e-100:
        return 1000000
    return 4*log10(prob/(1-prob))

piece_to_layer = {
    'R': 1,
    'N': 2,
    'B': 3,
    'Q': 4,
    'K': 5,
    'P': 6,
    'p': 7,
    'k': 8,
    'q': 9,
    'b': 10,
    'n': 11,
    'r': 12
}

def convert_to_bitboard(fen):
    boards = np.zeros((29,8,8), dtype=np.uint8)

    board = chess.Board(fen)
    turn_color = board.turn

    cr = board.castling_rights
    wkcastle = bool(cr & chess.BB_H1)
    wqcastle = bool(cr & chess.BB_A1)
    bkcastle = bool(cr & chess.BB_H8)
    bqcastle = bool(cr & chess.BB_A8)

    boards[0, :, :] = turn_color
    boards[25, :, :] = wkcastle
    boards[26, :, :] = wqcastle
    boards[27, :, :] = bkcastle
    boards[28, :, :] = bqcastle

    piece_map = board.piece_map()
    for i, p in piece_map.items():
        rank, file = divmod(i,8)
        layer = piece_to_layer[p.symbol()]
        boards[layer, rank, file] = 1

        for sq in board.attacks(i):
            attack_rank, attack_file = divmod(sq,8)
            boards[layer+12, attack_rank, attack_file] += 1 # could experiment with = 1 instead of += 1

    # print(piece_map)

    # print(list(boards))

    return boards

piece_to_material = {
    'R': 5,
    'N': 3,
    'B': 3,
    'Q': 9,
    'K': 0,
    'P': 1,
    'p': -1,
    'k': 0,
    'q': -9,
    'b': -3,
    'n': -3,
    'r': -5
}

# should carry material eval down the tree

def material_eval(board):
    pieces = board.piece_map()
    return sum([piece_to_material[piece.symbol()] for piece in pieces.values()])

def predict_model(fen):
    board = chess.Board(fen)

    if board.is_stalemate() or board.is_seventyfive_moves() or board.is_insufficient_material() or board.is_fivefold_repetition():
        return 0
    
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return prob_to_pawn_advantage(0)
        else:
            return prob_to_pawn_advantage(1)

    encoding = convert_to_bitboard(fen)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(29, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 128, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 128, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 256, kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Dropout(.5),
        torch.nn.Linear(256, 1)
    )

    model.load_state_dict(torch.load('conv_model_20.pt', map_location='cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with torch.no_grad():
        encoding = torch.Tensor(encoding)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(torch.unsqueeze(encoding, dim=0))
        # print(output.item(), convert_to_pawn_advantage(output.item()))

    return 50*(prob_to_pawn_advantage(output.item())+material_eval(board))

def tree_search(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return predict_model(board.fen())

    if maximizing_player:
        value = -1000000
        for move in board.legal_moves:
            board.push(move)
            value = max(value, tree_search(board, depth-1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = 1000000
        for move in board.legal_moves:
            board.push(move)
            value = min(value, tree_search(board, depth-1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

class ConvTestingBot:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        opening_books = ['vitamin17','Human','Titans','baron30']

        for book in opening_books:
            with chess.polyglot.open_reader(f'data/{book}.bin') as reader:
                board_hash = chess.polyglot.zobrist_hash(board)
                if reader.get(board_hash) is not None:
                    print(f'Found in {book} book.')
                    return reader.weighted_choice(board_hash).move

        print('Computing move.')

        depth = 2
        best_value = 1000000
        for move in board.legal_moves:
            board.push(move)
            value = tree_search(board, depth, -1000000, 1000000, True)
            board.pop()
            print(move, value)

            if value < best_value:
                best_value = value
                best_move = move
                
        print(best_move, best_value)
        return best_move


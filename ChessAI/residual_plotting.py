import chess
import chess.polyglot
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
from math import log10

def filter_mates(eval):
    if '#' in str(eval):
        if '+' in str(eval):
            return 20000
        if '-' in str(eval):
            return -20000
        else:
            return 0
    return int(eval)

def cut_extremes(eval):
    eval = max(eval, -200)
    eval = min(eval, 200)
    return eval

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

# piece_to_material = {
#     'R': 5,
#     'N': 3,
#     'B': 3,
#     'Q': 9,
#     'K': 0,
#     'P': 1,
#     'p': -1,
#     'k': 0,
#     'q': -9,
#     'b': -3,
#     'n': -3,
#     'r': -5
# }

# should carry material eval down the tree

# def material_eval(board):
#     pieces = board.piece_map()
#     return sum([piece_to_material[piece.symbol()] for piece in pieces.values()])

def predict_model(fen):
    board = chess.Board(fen)

    if board.is_stalemate() or board.is_seventyfive_moves() or board.is_insufficient_material() or board.is_fivefold_repetition():
        return 0
    
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return 1e-30
        else:
            return 1-1e-30

    encoding = convert_to_bitboard(fen)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(1856, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1)
    )

    model.load_state_dict(torch.load('models/linear_model_60.pt', map_location='cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with torch.no_grad():
        encoding = torch.Tensor(encoding)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(torch.unsqueeze(encoding, dim=0))
        # print(output.item(), convert_to_pawn_advantage(output.item()))

    return 100*(prob_to_pawn_advantage(output.item()))

if __name__ == '__main__':
    df = pd.read_csv('data/smallestChessData.csv')
    df['EvalPredictions'] = df['FEN'].apply(predict_model).apply(cut_extremes)
    df['Evaluation'] = df['Evaluation'].apply(filter_mates).apply(cut_extremes)

    plt.scatter(df['EvalPredictions'], df['Evaluation'])
    plt.savefig('residual_plot.png')

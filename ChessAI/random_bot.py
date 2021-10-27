import chess
import random

class RandomBot:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        moves = list(board.legal_moves)
        return moves[int(len(moves) * random.random())]
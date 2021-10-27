import chess

class CarissaBot:
    def __init__(self, board):
        self.board = board

    def get_move(self):
        legal_moves = self.board.legal_moves

    def evaluate_board(self, board):
        print(board)



board = chess.Board()

moves = board.legal_moves

print(moves)

board.push_san("Nh3")

write_path = 'current_board.svg'
open(write_path, 'w').write(chess.svg.board(board, size=350))
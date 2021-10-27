import chess
import chess.polyglot
import random

class TestingBot:
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

        print("Random Move")
        moves = list(board.legal_moves)
        return moves[int(len(moves) * random.random())]

        
import chess
import chess.polyglot
import random
import numpy as np

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

        depth = 3
        best_value = -1000000
        for move in board.legal_moves:
            board.push(move)
            value = self.tree_search(board, depth, -1000000, 1000000)
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

        # print("Random Move")
        # moves = list(board.legal_moves)
        # return moves[int(len(moves) * random.random())]

    def tree_search(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.material_evaluate(board)

        best_value = -1000000
        for move in board.legal_moves:
            board.push(move)
            value = -self.tree_search(board, depth - 1, -beta, -alpha)
            board.pop()
            if value > best_value:
                best_value = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best_value

    def material_evaluate(self, board):
        if board.is_checkmate():
            return -1000000 if board.turn == self.color else 1000000

        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0

        material = 0

        type_pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        value_pieces = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10000}
        color_pieces = np.array([chess.WHITE, chess.BLACK])
        all_pieces = []
        for type_piece in type_pieces:
            for color_piece in color_pieces:
                all_pieces.append((type_piece, color_piece))

        for piece, color in all_pieces:
            if color == chess.WHITE:
                material += len(board.pieces(piece, color)) * value_pieces[piece]
            if color == chess.BLACK:
                material -= len(board.pieces(piece, color)) * value_pieces[piece]

        return material
        
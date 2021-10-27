import chess

class PlayerBot():
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        while True:
            move = input("Enter move in SAN notation: ")
            try:
                board.parse_san(move)
            except:
                print("Invalid move. Try again.")
                continue
            break
            
        return board.parse_san(move)

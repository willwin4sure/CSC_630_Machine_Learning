""" This file corresponds to the problem "Tic Tac Toe" the Python Crash
Course.

"""

# imports go here if you need any, e.g.:
# import random
from statistics import Counter
import json

# global constants go here if you need any, e.g.:
# NUMBER_OF_BOARDS = 200

class TicTacToe:
    """ Tic-Tac-Toe board with some methods of determining if the game is over, whose turn it is, and who won.

    Attributes:
        game_id: An int containing the ID of the Tic-Tac-Toe game
        board: A 2-dimensional list describing the state of the board, with 'X', 'O', and None as entries

    """
    def __init__ (self,board):
        self.game_id = int(board[0:3])
        self.board = self.make_board(board)

    def make_board(self, board_string):
        """
        Create the board, a 2-dimensional list.

        Params:
            board_string -- multi-line string containing the board description,
                including as its first line the game_id

        Return:
            a 2-dimensional list describing the state of the board.
        """
        lines = board_string.split('\n')
        board = []
        for i in range(3):
            cells = lines[2*i+1].split('|')
            row = []
            for j in range(3):
                if 'X' in cells[j]:
                    row.append('X')
                elif 'O' in cells[j]:
                    row.append('O')
                else:
                    row.append(None)
            board.append(row)
        return board

    def is_game_over(self):
        """
        Determine if the game is over.

        Return:
            1 if X has won, -1 if O has won, 2 if tie, 0 otherwise
            (Or use whatever you want to use to describe the condition.
            Methods with a name that sounds like a question usually can get
            called from an if statement, and this one can:
            >>> if my_game.is_game_over():
            >>>     do_something()
            Because 0 evaluates to `False`, and non-zero to `True`.
            )
        """

        def convert(string):
            if string == 'X':
                return 1
            if string == 'O':
                return -1
            return 0

        for i in range(3):
            if (len(set(self.board[i])) == 1):
                return convert(self.board[i][0])
            if (len(set([self.board[j][i] for j in range(3)])) == 1):
                return convert(self.board[0][i])
            
        if len(set([self.board[i][i] for i in range(3)])) == 1:
            return convert(self.board[0][0])

        if len(set([self.board[i][2-i] for i in range(3)])) == 1:
            return convert(self.board[0][2])
        
        if (None in self.board[0] or None in self.board[1] or None in self.board[2]):
            return 0

        return 2

    def determine_turn(self):
        """
        Determine whose turn it is, if possible.

        This method assumes that the game is not over.

        Return:
            1 if X's turn, -1 if O's turn, 0 otherwise
        """
        nums = Counter()
        for i in range(3):
            for j in range(3):
                nums[self.board[i][j]] += 1
        
        if (nums[None] == 0):
            return 0
        
        if (nums['X'] == nums['O'] + 1):
            return -1
        
        if (nums['X'] == nums['O']):
            return 1
        
        return 0

if __name__ == "__main__":
    """ This code block is for your script, namely part 4.  What the above
    `if` statement checks is the following: if I called this script from
    the command line, then execute this code block; otherwise skip it.

    The point is this: suppose you wanted to import your `TicTacToe` class
    into another file.  Then you probably wouldn't want to have this code
    block execute, because for example you may not even have a text file
    called `tictactoe_games.txt` present.
    """
    # Look up this "with" keyword, and understand what it does.
    with open("tictactoe_games.txt",'r') as f_in:
        lines = f_in.readlines()

        data = {}
        data['games'] = []
        xo_dict = {1: 'X', -1: 'O', 2: 'T'}
        for i in range(len(lines)//7):
            single_board = ''.join(lines[7*i+1:7*i+7])
            single_game = TicTacToe(single_board)
            if single_game.is_game_over():
                # print(f'game_id: {str(single_game.game_id).zfill(3)} complete: True Victor: {xo_dict[single_game.is_game_over()]}')
                data['games'].append({'game_id' : str(single_game.game_id).zfill(3),\
                                      'complete' : True,\
                                      'victor' : xo_dict[single_game.is_game_over()]})
            else:
                # print(f'game_id: {str(single_game.game_id).zfill(3)} complete: False Turn: {xo_dict[single_game.determine_turn()]}')
                data['games'].append({'game_id' : str(single_game.game_id).zfill(3),\
                                      'complete' : False,\
                                      'victor' : xo_dict[single_game.determine_turn()]})
        with open('data.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
                
        # implement me!
    # example_board = ("001)\n"
    #                     " X | O | X \n"
    #                     "-----------\n"
    #                     " O |   | X \n"
    #                     "-----------\n"
    #                     "   |   | O")
    # example_game = TicTacToe(example_board)
    # print(example_game.game_id)
    # print(example_game.board)
    # print(example_game.is_game_over())
    # print(example_game.determine_turn())

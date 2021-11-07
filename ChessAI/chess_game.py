import chess
import chess.svg

import player_bot
import random_bot
import testing_bot
import testing_bot_conv

if __name__ == "__main__":
    '''Runs a while loop to play out the chess game between two of the bots, player bot and random bot,
    by prompting for their get_move methods alternatingly and calling them on the current board state,
    then pushing the move onto the board and saving the current board state to an svg file called current_board.svg,
    and ending the loop whenever a player gets checkmated.'''

    board = chess.Board()
    player1 = player_bot.PlayerBot(chess.WHITE)
    player2 = testing_bot_conv.ConvTestingBot(chess.BLACK)
    while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves()):
        if board.turn == chess.WHITE:
            player_move = player1.get_move(board)
            board.push(player_move)
        else:
            player_move = player2.get_move(board)
            board.push(player_move)
        write_path = f'current_board.svg'
        with open(write_path, 'w') as f:
            f.write(chess.svg.board(board, size=350))
        print(board.fen())
    print('Game over!')
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            print('Black wins!')
        else:
            print('White wins!')
    else:
        print('Draw!')
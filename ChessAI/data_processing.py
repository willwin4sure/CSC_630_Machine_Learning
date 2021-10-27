import pandas as pd
import chess

def convert_fen_to_encoding(fen_string):
    one_hot_dict = {'P': '10100000', 'N': '10010000', 'B': '10001000', 'R': '10000100', 'Q': '10000010', 'K': '10000001', 'p': '01100000', 'n': '01010000', 'b': '01001000', 'r': '01000100', 'q': '01000010', 'k': '01000001', '.': '00000000'}
    fen_string_props = fen_string.split(' ')
    rows = chess.Board(fen_string).__str__().split('\n')
    squares_encoding = []
    for row in rows:
        squares_encoding.append(list(map(lambda x: one_hot_dict[x], row.split(' '))))

    if fen_string_props[1] == 'w':
        turn = '10'
    elif fen_string_props[1] == 'b':
        turn = '01'
    else:
        turn = '00'

    castle_privileges = fen_string_props[2]
    castle_privileges_encoding = ''.join([str(int(x in castle_privileges)) for x in ['K', 'Q', 'k', 'q']])

    row_encoding = []
    for row in squares_encoding:
        row_encoding.append(''.join(row))

    board_encoding = ''.join(row_encoding)
    full_encoding = ''.join([board_encoding, turn, castle_privileges_encoding])
    
    return full_encoding

df = pd.read_csv('data/smallerChessData.csv')

df['Encoding'] = df['FEN'].apply(convert_fen_to_encoding)

def filter_mates(eval):
    if '#' in str(eval):
        if '+' in str(eval):
            return 20000
        if '-' in str(eval):
            return -20000
        else:
            return 0
    return int(eval)

df['EvaluationFiltered'] = df['Evaluation'].apply(filter_mates)

with open('data/smallerChessDataEncoded.csv', 'w') as f:
    df.to_csv(f, index=False)

print(df.head())

# df['FEN'] = df['FEN'].apply(lambda x: chess.Board(x))

# print(df['FEN'].iloc[0])


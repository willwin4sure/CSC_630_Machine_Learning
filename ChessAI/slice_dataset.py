import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_smaller = df.sample(frac=0.08)

df_smaller_white = df_smaller[df_smaller['FEN'].apply(lambda x: x.split(' ')[1]) =='w']

with open('data/smallerChessData2White.csv', 'w') as f:
    df_smaller_white.to_csv(f, index=False)
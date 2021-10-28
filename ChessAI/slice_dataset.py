import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_smaller = df.iloc[0:1000000]

with open('data/smallChessData.csv', 'w') as f:
    df_smaller.to_csv(f, index=False)
import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_smaller = df.iloc[0:500000]

with open('data/smallerChessData.csv', 'w') as f:
    df_smaller.to_csv(f, index=False)
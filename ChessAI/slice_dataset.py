import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_smaller = df.sample(frac=0.08)

with open('data/smallChessData2.csv', 'w') as f:
    df_smaller.to_csv(f, index=False)
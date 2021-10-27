import pandas as pd

df = pd.read_csv("data/chessData.csv")

def filter_hashes(eval):
    if '#' in str(eval):
        return int(eval[1:])
    return int(eval)

print(df.head())

df['EvaluationNum'] = df['Evaluation'].apply(filter_hashes)
print(df.head())
print(df['EvaluationNum'].max())
print(df['EvaluationNum'].min())


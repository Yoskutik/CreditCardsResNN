import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/creditcard.csv')
df = df.drop('Time', axis=1)

df_train, df_test = train_test_split(df, train_size=0.2, stratify=df.Class)

df_test = df_test.sort_values(by=['Class'], ascending=False)
df_test = df_test.iloc[:(df_test.Class == 1).sum() * 2]

df_train = df_train.sort_values(by=['Class'], ascending=False)
df_train = df_train.iloc[:(df_train.Class == 1).sum() * 4]

df_train.to_csv('data/train.csv', index=False)
df_test.to_csv('data/test.csv', index=False)

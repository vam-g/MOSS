import pandas as pd

df_train = pd.read_csv('../data/rlhf_step3/train.csv')[:1000]
df_eval = pd.read_csv('../data/rlhf_step3/eval.csv')[:1000]
df_train.to_csv('../data/rlhf_step3_1000samples/train.csv',index=False)
df_eval.to_csv('../data/rlhf_step3_1000samples/eval.csv',index=False)
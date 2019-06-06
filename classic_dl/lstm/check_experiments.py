import numpy as np
import pandas as pd
# pd.set_option('display.max_colums', 15)
# pd.set_option('display.max_rows', 10)

df = pd.read_csv("experiments_lstm.csv")

val_acc = 0.9
val_roc_auc = 0.9

subdf = df[(df['acc'] >= val_acc) & (df['roc-auc'] >= val_roc_auc)]
# subdf = df[(df['acc'] >= val_acc) & (df['roc-auc'] >= val_roc_auc) & (df['target_steps_ahead'] == 2000]

print(subdf)

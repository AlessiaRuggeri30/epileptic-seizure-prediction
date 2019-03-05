import numpy as np
import pandas as pd

df = pd.read_csv("experiments_lstm.csv")

val_acc = 0.9
val_roc_auc = 0.9

subdf = df[(df['acc'] >= val_acc) & (df['roc-auc'] >= val_roc_auc)]

print(subdf)

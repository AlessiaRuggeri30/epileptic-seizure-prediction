# -----------------------------------------------------------------------------
# Manage experiments
# -----------------------------------------------------------------------------
df=df.rename(columns = {'Unnamed: 0': ''})
df = df.set_index('')
df.to_pickle(experiments_lstm)
df.to_csv('experiments_lstm.csv')

# Check experiments
import pandas as pd
df = pd.read_csv("experiments_conv_pred_dilated.csv")
val_acc = 0.9
val_roc_auc = 0.85
subdf = df[(df['acc'] >= val_acc) & (df['roc-auc'] >= val_roc_auc)]
subdf = df[(df['acc'] >= val_acc) & (df['roc-auc'] >= val_roc_auc) & (df['target_steps_ahead'] == 2000)]
print(subdf)

# -----------------------------------------------------------------------------
# Cross validation results CONVOLUTIONAL
# -----------------------------------------------------------------------------
# Mean of results of model applied to 3 folds
import pandas as pd
filename = "experiments_conv_det_final.csv"
df = pd.read_csv(filename)
df = df[67:]
columns = ['epochs', 'depth_conv', 'depth_dense', 'filters', 'kernel_size',
           'activation', 'l2_reg', 'batch_norm', 'dropout', 'pooling', 'pool_size', 'padding',
           'dilation_rate', 'stride', 'subsampling_factor', 'look_back',
           'target_steps_ahead']
treshold = 0.60
group_df = df.groupby(columns)['loss', 'acc', 'roc-auc', 'recall'].agg('mean')
group_df = group_df[group_df['recall'] >= treshold]
print(group_df)

print(df[(df['depth_conv'] == 3) & (df['stride'] == 5) & (df['look_back'] == 200) & (df['target_steps_ahead'] == 1000)])

# -----------------------------------------------------------------------------
# Cross validation results LSTM
# -----------------------------------------------------------------------------
import pandas as pd
filename = "experiments_lstm_det_final.csv"
df = pd.read_csv(filename)
columns = ['epochs', 'depth_lstm', 'depth_dense', 'units_lstm',
           'activation', 'l2_reg', 'batch_norm', 'dropout',
           'stride', 'subsampling_factor', 'look_back',
           'target_steps_ahead']
treshold = 0.63
group_df = df[df['epochs'] == 15]
group_df = group_df.groupby(columns)['loss', 'acc', 'roc-auc', 'recall'].agg('mean')
group_df = group_df[group_df['recall'] >= treshold]
print(group_df)

print(df[(df['epochs'] == 15) & (df['stride'] == 1) & (df['look_back'] == 200) & (df['target_steps_ahead'] == 500)])

# -----------------------------------------------------------------------------
# Cross validation results DENSE
# -----------------------------------------------------------------------------
import pandas as pd
filename = "experiments_dense_det_final.csv"
df = pd.read_csv(filename)
columns = ['epochs', 'units', 'activation', 'l2_reg', 'batch_norm', 'dropout']
group_df = df.groupby(columns)['loss', 'acc', 'roc-auc', 'recall'].agg('mean')
print(group_df)

print(df[(df['epochs'] == 15) & (df['stride'] == 1) & (df['look_back'] == 200) & (df['target_steps_ahead'] == 500)])

# -----------------------------------------------------------------------------
# Cross validation results GRAPH-CONVOLUTIONAL
# -----------------------------------------------------------------------------
import pandas as pd
filename = "experiments_conv_pred.csv"
df = pd.read_csv(filename)
columns = ['epochs', 'depth_conv', 'depth_dense', 'filters', 'kernel_size', 'g_filters',
           'activation', 'l2_reg', 'batch_norm', 'dropout', 'pooling', 'pool_size', 'padding',
           'dilation_rate', 'stride', 'subsampling_factor', 'samples_per_graph', 'look_back',
           'target_steps_ahead']
# treshold = 0.63
group_df = group_df.groupby(columns)['loss', 'acc', 'roc-auc', 'recall'].agg('mean')
# group_df = group_df[group_df['recall'] >= treshold]
print(group_df)

print(df[(df['epochs'] == 15) & (df['stride'] == 1) & (df['look_back'] == 200) & (df['target_steps_ahead'] == 500)])
print(df[(df['roc-auc'] >= 0.60)])

# -----------------------------------------------------------------------------
# Cross validation results GRAPH-LSTM
# -----------------------------------------------------------------------------
import pandas as pd
filename = "experiments_lstm_pred.csv"
df = pd.read_csv(filename)
columns = ['epochs', 'depth_lstm', 'depth_dense', 'units_lstm', 'g_filters',
           'activation', 'l2_reg', 'batch_norm', 'dropout',
           'stride', 'subsampling_factor', 'samples_per_graph', 'look_back',
           'target_steps_ahead']
# treshold = 0.63
group_df = group_df.groupby(columns)['loss', 'acc', 'roc-auc', 'recall'].agg('mean')
# group_df = group_df[group_df['recall'] >= treshold]
print(group_df)

print(df[(df['epochs'] == 15) & (df['stride'] == 1) & (df['look_back'] == 200) & (df['target_steps_ahead'] == 500)])
print(df[(df['roc-auc'] >= 0.60)])


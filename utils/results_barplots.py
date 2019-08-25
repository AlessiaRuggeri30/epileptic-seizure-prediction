import numpy as np
import matplotlib.pyplot as plt

''' Global parameters '''
N = 4       # num of metrics (loss, acc, roc-auc, recall)
ind = np.arange(N)      # the x locations for the metrics
width = 0.2        # the width of the bars

''' File name and path '''
saving_name = "pred_sequence2000_plots"
saving_path = f"./results_plots/{saving_name}.png"

''' Metrics parameters '''
# The length of each list should be always N, while the number of lists corresponds to the number of models
model1_vals = [0.236, 0.940, 0.911, 0.604]
model2_vals = [0.204, 0.956, 0.932, 0.767]
model3_vals = [2.549, 0.328, 0.642, 0.790]
model4_vals = [2.103, 0.467, 0.475, 0.463]
y_max = 2.75
y_step = 0.25

''' Models names '''
model1 = 'CNN'
model2 = 'LSTM'
model3 = 'graph-based CNN'
model4 = 'graph-based LSTM'

''' Figure generation '''
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, model1_vals, width, color='mediumseagreen')
rects2 = ax.bar(ind+width, model2_vals, width, color='orange')
rects3 = ax.bar(ind+width*2, model3_vals, width, color='cornflowerblue')
rects4 = ax.bar(ind+width*3, model4_vals, width, color='lightcoral')
ax.axhline(y=1, color='silver', linestyle='--', zorder=0)

''' Labels generation '''
names = (model1, model2, model3, model4)
rects_labels = (rects1[0], rects2[0], rects3[0], rects4[0])

ax.set_ylabel('Scores')
ax.set_xticks(ind+(1/N))
ax.set_xticklabels(('Loss', 'Accuracy', 'ROC-AUC', 'Recall'))
plt.yticks(np.arange(0, y_max+0.1, y_step))
ax.legend(rects_labels, names)
plt.tight_layout()

plt.savefig(saving_path)

# -----------------------------------------------------------------------------
# MODELS PARAMETERS
# -----------------------------------------------------------------------------

# DETECTION ON A TIME STEP
# ''' File name and path '''
# saving_name = "det_timestep_plots"
# saving_path = f"./results_plots/{saving_name}.png"
#
# ''' Metrics parameters '''
# # The length of each list should be always N, while the number of lists corresponds to the number of models
# model1_vals = [2.497, 0.751, 0.601, 0.067]
# model2_vals = [0.303, 0.899, 0.768, 0.121]
# model3_vals = [0.297, 0.898, 0.805, 0.219]
# model4_vals = [0.293, 0.902, 0.734, 0.248]
# y_max = 2.75
# y_step = 0.25
#
# ''' Models names '''
# model1 = 'SVM'
# model2 = 'random forest'
# model3 = 'gradient boosting'
# model4 = 'FCNN'


# DETECTION ON A SEQUENCE
# ''' File name and path '''
# saving_name = "det_sequence_plots"
# saving_path = f"./results_plots/{saving_name}.png"
#
# ''' Metrics parameters '''
# # The length of each list should be always N, while the number of lists corresponds to the number of models
# model1_vals = [0.208, 0.942, 0.903, 0.534]
# model2_vals = [0.221, 0.953, 0.910, 0.580]
# model3_vals = [2.268, 0.297, 0.667, 0.895]
# model4_vals = [1.484, 0.692, 0.616, 0.476]
# y_max = 2.5
# y_step = 0.25
#
# ''' Models names '''
# model1 = 'CNN'
# model2 = 'LSTM'
# model3 = 'graph-based CNN'
# model4 = 'graph-based LSTM'


# PREDICTION ON A SEQUENCE 500
# ''' File name and path '''
# saving_name = "pred_sequence500_plots"
# saving_path = f"./results_plots/{saving_name}.png"
#
# ''' Metrics parameters '''
# # The length of each list should be always N, while the number of lists corresponds to the number of models
# model1_vals = [0.340, 0.892, 0.946, 0.634]
# model2_vals = [0.293, 0.922, 0.922, 0.708]
# model3_vals = [1.812, 0.291, 0.570, 0.877]
# model4_vals = [1.532, 0.674, 0.558, 0.377]
# y_max = 2
# y_step = 0.25


# PREDICTION ON A SEQUENCE 1000
# ''' File name and path '''
# saving_name = "pred_sequence1000_plots"
# saving_path = f"./results_plots/{saving_name}.png"
#
# ''' Metrics parameters '''
# # The length of each list should be always N, while the number of lists corresponds to the number of models
# model1_vals = [0.224, 0.924, 0.930, 0.702]
# model2_vals = [0.180, 0.954, 0.908, 0.637]
# model3_vals = [2.990, 0.329, 0.661, 0.779]
# model4_vals = [1.524, 0.614, 0.547, 0.439]
# y_max = 3
# y_step = 0.25


# PREDICTION ON A SEQUENCE 2000
''' File name and path '''
# saving_name = "pred_sequence2000_plots"
# saving_path = f"./results_plots/{saving_name}.png"
#
# ''' Metrics parameters '''
# # The length of each list should be always N, while the number of lists corresponds to the number of models
# model1_vals = [0.236, 0.940, 0.911, 0.604]
# model2_vals = [0.204, 0.956, 0.932, 0.767]
# model3_vals = [2.549, 0.328, 0.642, 0.790]
# model4_vals = [2.103, 0.467, 0.475, 0.463]
# y_max = 2.75
# y_step = 0.25


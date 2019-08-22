import numpy as np
import matplotlib.pyplot as plt

''' Global parameters '''
N = 4       # num of metrics (loss, acc, roc-auc, recall)
ind = np.arange(N)      # the x locations for the metrics
width = 0.2        # the width of the bars
saving_name = "ml_models"
saving_path = f"./results_plots/{saving_name}.png"

''' Metrics parameters '''
# The length of each list should be always N, while the number of lists corresponds to the number of models
model1_vals = [0.3, 0.9, 0.77, 0.12]
model2_vals = [0.3, 0.9, 0.8, 0.22]
model3_vals = [2.5, 0.75, 0.6, 0.07]
model4_vals = [0.29, 0.9, 0.73, 0.25]
y_max = 2.75      # 3
y_step = 0.25      # 0.25

''' Models names '''
model1 = 'random forest'
model2 = 'gradient boosti ng'
model3 = 'SVM'
model4 = 'FCNN'

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

# def autolabel(rects, vals):
#     for i in range(len(rects)):
#         h = vals[i]
#         ax.text(rects[i].get_x()+rects[i].get_width()/2., 1.05*h, '%s'%h,
#                 ha='center', va='bottom')
#
# autolabel(rects1, model1_vals)
# autolabel(rects2, model2_vals)
# autolabel(rects3, model3_vals)
# autolabel(rects4, model4_vals)
plt.tight_layout()

plt.savefig(saving_path)
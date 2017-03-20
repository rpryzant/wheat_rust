"""
=== DESCRIPTION
Given the output of a model run, plots the saved roc curve


python roc_plotter.py best_conv baseline_spectral baseline_hist

=== USAGE
   NOTE: only the first curve will be plotted
"""
import sys
import matplotlib.pyplot as plt
import json


def extract_roc(s):
    try:
        out_data = json.loads(s)
    except:
        out_data = eval(s)

    return eval(out_data['result']['rocs']), out_data['result']['auc']


[fpr, tpr, thresholds], auc = extract_roc(next(open(sys.argv[1])))

[fpr_base, tpr_base, thresholds], auc = extract_roc(next(open(sys.argv[2])))

[fpr_hist, tpr_hist, thresholds], auc = extract_roc(next(open(sys.argv[3])))


print 'AUC: ', auc
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='deep features')
plt.plot(fpr_base, tpr_base, label="spectral features")
plt.plot(fpr_hist, tpr_hist, label="raw histogram features")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



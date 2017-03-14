"""
=== DESCRIPTION
Given the output of a model run, plots the saved roc curve

=== USAGE
python roc_plotter.py [outfile]
   NOTE: only the first curve will be plotted
"""
import sys
import matplotlib.pyplot as plt
import json


out = next(open(sys.argv[1]))
try:
    out_data = json.loads(out)
except:
    out_data = eval(out)

curve = eval(out_data['result']['rocs'])

[fpr, tpr, thresholds] = curve




plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Best deep features')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



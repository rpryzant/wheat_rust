import csv
import matplotlib.pyplot as plt
import os



with open('month_results.csv','r') as csvFile:
    reader=csv.reader(csvFile,delimiter=',')
    next(reader)
    d = {r[0]: r[1:] for r in reader}

plt.figure(1)
for line_name, line in d.iteritems():
    plt.plot(range(len(line)), line, label=line_name)

plt.xlabel('Predicting Month')
plt.ylabel('AUC')
plt.title('July Aug Sep Oct Nov Dec Jan Feb Mar Apr ')
plt.legend(loc='best')
plt.show()










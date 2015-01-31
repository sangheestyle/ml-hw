# see http://goo.gl/AwIS2n
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

csv_path = 'result.csv'
result = defaultdict(dict)
with open(csv_path, 'rb') as f:
    reader = csv.DictReader(f)
    for row in reader:
        result[int(row['n'])][int(row['k'])] = float(row['accuracy'])

fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
for n in sorted(result.keys(), reverse=True):
    k = sorted(result[n].keys())
    accuracy = [result[n][i] for i in k]
    ax.plot(k, accuracy, label="n: " + str(n), marker='x')

ax.legend(loc=1); # upper right corner
ax.set_xlabel('k: the number of nearest neighbors')
ax.set_ylabel('accuracy')
ax.set_xlim(0, 10)
ax.set_ylim(0.55, 0.87)
ax.grid(True)
ax.set_title('Accuracy trend: "n" is the number of training points');
fig.savefig('accuracy.png', dpi=100)

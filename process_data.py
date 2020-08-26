import os
import numpy as np
import pandas as pd

out_f = open('all_data.csv', 'w+')

with open('ppdb_all.txt', 'r+') as f:
    for line in f:
        line = line.split('|||')
        if float(line[-1]) >= 3.0:
            out_f.write('{},{}\n'.format(line[0], line[1]))

data = pd.read_csv('quora_duplicate_questions.tsv', sep="\t") 

data = np.array(data)
data = data[data[:,-1]==1] # only collect true paraphrases

for row in data:
    out_f.write('{},{}\n'.format(row[3], row[4]))


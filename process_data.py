import os
import numpy as np
import pandas as pd

out_f = open('quora.csv', 'w+')

data = pd.read_csv('quora_duplicate_questions.tsv', sep="\t") 

data = np.array(data)
data = data[data[:,-1]==1]
print(data.shape)

for row in data:
    out_f.write('{},{}\n'.format(row[3], row[4]))
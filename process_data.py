import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

out_f_train = open('train_data.csv', 'w+')
out_f_val = open('val_data.csv', 'w+')

all_source = []
all_target = []

with open('ppdb_all.txt', 'r+') as f:
    for line in f:
        line = line.split('|||')
        if float(line[-1]) >= 3.0:
            all_source.append(line[0])
            all_target.append(line[1])

data = pd.read_csv('quora_duplicate_questions.tsv', sep="\t") 
data = np.array(data)
data = data[data[:,-1]==1] # only collect true paraphrases

for row in data:
    all_source.append(row[-3])
    all_target.append(row[-2])

all_data = list(zip(all_source, all_target))
np.random.shuffle(all_data)

source_train, source_val, target_train, target_val = train_test_split([x[0] for x in all_data], [x[1] for x in all_data], test_size=0.3)

for i in range(len(source_train)):
    out_f_train.write('{},{}\n'.format(source_train[i], target_train[i]))

for i in range(len(source_val)):
    out_f_val.write('{},{}\n'.format(source_val[i], target_val[i]))
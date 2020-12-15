import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def collect_ppdb():
    sources = []
    targets = []
    with open('ppdb_all.txt', 'r+') as f:
        for line in f:
            line = line.split('|||')
            if float(line[-1]) >= 3.0:
                sources.append(line[0])
                targets.append(line[1])

    return sources, targets

def collect_quora():
    sources = []
    targets = []

    data = pd.read_csv('quora_duplicate_questions.tsv', sep="\t") 
    data = np.array(data)
    data = data[data[:,-1]==1] # only collect true paraphrases

    for row in data:
        sources.append(row[-3])
        targets.append(row[-2])
    
    return sources, targets 

def collect_language_net():
    sources = []
    targets = []

    with open('2016_Oct_10--2017_Jan_08_paraphrase.txt', 'r+') as f:
        for line in f:
            line = line.split('\t')
            if len(line) == 2:
                sources.append(line[0].strip())
                targets.append(line[1].strip())

    return sources, targets 

def save_to_file(out_file, sources, targets):
    for i in range(len(sources)):
        out_file.write('{},{}\n'.format(sources[i], targets[i]))
    out_file.close()

if __name__ == '__main__':
    out_f_train = open('train_data_all.csv', 'w+')
    out_f_val = open('val_data_all.csv', 'w+')
    out_f_test = open('test_data_all.csv', 'w+')

    ppdb_sources, ppdb_targets = collect_ppdb()
    quora_sources, quora_targets = collect_quora()
    ln_sources, ln_targets = collect_language_net()

    all_data = list(zip(ppdb_sources + quora_sources + ln_sources, ppdb_targets + quora_targets + ln_targets))

    source_train, source_val, target_train, target_val = train_test_split([x[0] for x in all_data], [x[1] for x in all_data], test_size=0.05)

    source_val, source_test, target_val, target_test = train_test_split(source_val, target_val, test_size=0.2) 

    save_to_file(out_f_train, source_train, target_train)
    save_to_file(out_f_val, source_val, target_val)
    save_to_file(out_f_test, source_test, target_test)
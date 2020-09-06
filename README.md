# Paraphrase-Generation

This repository explores sequence-to-sequence paraphrase generation inspired by neural machine translation. 

Install required packages
'''
pip3 install -r requirements.txt
'''

Collect data into csv files in the format: first column - source sentences, second columns - target (paraphrased) sentences.
Datasets used include PPDB (http://paraphrase.org/#/download) and Quora Question Pairs (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs).

'''
python3 process_data.py
''' 

Create training vocabulary and save it into a json file
'''
python3 construct_vocabulary.py --data_path train_data.csv --vocab_path vocab.json
'''

Perform training
'''
python3 train.py --train_path train_data.csv --val_path val_data.csv --device cpu
'''
 

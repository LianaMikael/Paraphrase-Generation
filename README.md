# Paraphrase-Generation

This repository explores sequence-to-sequence paraphrase generation inspired by early neural machine translation models. This is a natural language generation task where for a given text, the objective is to generate a semantically similar sentence that is expressed differently. The end-to-end architecture consists of a bidirectional LSTM encoder, a unidirectional LSTM decoder and a global attention mechanism. I explore both word-level embeddings as well as byte pair encoding. 
Automatic evaluation includes both word overlap-based methods as well as embedding-based metrics. Word overlap-based metrics focus on evaluating word overlap between predicted sentences and target sentences (BLEU score, word error rate). Embedding-based metrics consider meanings of sentences by combining word embeddings (Word2Vec, Glove) and compute a distance measure (cosine distance) between embedding vectors of predicted and target sentences.   

Install required packages
```
pip3 install -r requirements.txt
```

Collect data into csv files in the format: first column - source sentences, second columns - target (paraphrased) sentences.
Datasets used include PPDB (http://paraphrase.org/#/download) and Quora Question Pairs (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs).

```
python3 process_data.py
``` 

Create training vocabulary and save it into a json file
```
python3 construct_vocabulary.py --data_path train_data.csv --vocab_path vocab.json
```

Perform training
```
python3 train.py --train_path train_data.csv --val_path val_data.csv --device cpu
```

Perform evaluation 
'''
python3 evaluate.py --test_path test_data.csv --device cpu
''' 

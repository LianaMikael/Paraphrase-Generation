# Paraphrase-Generation

This repository explores sequence-to-sequence paraphrase generation inspired by early neural machine translation models. This is a natural language generation task where for a given text, the objective is to generate a semantically similar sentence that is expressed differently. 

The end-to-end architecture consists of a bidirectional LSTM encoder, a unidirectional LSTM decoder and a global attention mechanism. I explore word-level encoding. Byte pair encoding can also be used. 

## Architecture and Training Procedure

For a source sentence from the training set, we look up word embeddings from the embeddings matrix, obtaining fixed-dimensional vectors. These embedding vectors are then fed into the bidirectional LSTM, producing hidden states and cell states for both the forward and backward LSTMs. We concatenate them as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\\h_i^{enc}&space;=&space;[h_i^{\leftarrow},&space;h_i^{\rightarrow}]\\&space;c_i^{enc}&space;=&space;[c_i^{\leftarrow},&space;c_i^{\rightarrow}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\h_i^{enc}&space;=&space;[h_i^{\leftarrow},&space;h_i^{\rightarrow}]\\&space;c_i^{enc}&space;=&space;[c_i^{\leftarrow},&space;c_i^{\rightarrow}]" title="\\h_i^{enc} = [h_i^{\leftarrow}, h_i^{\rightarrow}]\\ c_i^{enc} = [c_i^{\leftarrow}, c_i^{\rightarrow}]" /></a>  

The purpose of these vectors is to encapsulate information about the sentence so that the decoder is be able to accurately generate the output sentence. We then perform a linear projection of the hidden and cell state and use the outputs to initialize decoder's initial hiidden and cell states.

During decoding, at each time step, we look up the embeddings of the current target word and contcatenate it with the previous combined output vector (initalized as a vector of zeros).

Multiplicative attention is calculated by applying a projection layer to the encoder's hidden state, multiplying by the current hidden state of the decoder, applying a softmax function to get attention probability distribution and multiplying it with encoder's hidden state: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\\e_{t,i}&space;=&space;(h^{dec}_t)^TWh_i^{enc}&space;\\p_t=softmax(e_t)&space;\\a_t&space;=&space;\sum_{i=1}^{m}p_{t,i}h_i^{enc}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\e_{t,i}&space;=&space;(h^{dec}_t)^TWh_i^{enc}&space;\\p_t=softmax(e_t)&space;\\a_t&space;=&space;\sum_{i=1}^{m}p_{t,i}h_i^{enc}" title="\\e_{t,i} = (h^{dec}_t)^TWh_i^{enc} \\p_t=softmax(e_t) \\a_t = \sum_{i=1}^{m}p_{t,i}h_i^{enc}" /></a>


Finally, we concatenate the attention output vector with decoder's hidden state and pass it through a linear layer, tanh and dropout:

<a href="https://www.codecogs.com/eqnedit.php?latex=\\u_t=[a_t,h_t^{dec}]&space;\\v_t=W&space;u_t&space;\\o_t=dropout(tanh(v_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\u_t=[a_t,h_t^{dec}]&space;\\v_t=W&space;u_t&space;\\o_t=dropout(tanh(v_t))" title="\\u_t=[a_t,h_t^{dec}] \\v_t=W u_t \\o_t=dropout(tanh(v_t))" /></a>

This gives us the combined output vector for the current time step. To compute the probability distribution over the target words, we apply a softmax. This is then passed through softmax cross entropy along with the target word's vector to obtain loss for training. 

## Output Sentence Generation

There are three main ways to generate output:  
- **Greedy Decoding**: at each time step, select the word from the vocabulary with the highst score obtained by performing one decoder step, continue until the end token is reached or a pre-defined maximum number of steps is completed.  
- **Beam Search**: at each time step, keep track of *k* (beam size) most probable hypotheses, for each hypothesis continue until the end token is reached, stop the process when a maximum number of steps is reached or a pre-defined minimum number of hypotheses is generated.  
- **Random Sampling**: randomly choose an output sequence. This technique may be particulary useful for cases when we wish to have a variety of outputs for one input, for example for seq2seq systems such as dialogues (not implemented here).

## Evaluation
 
Automatic evaluation can be separated into two clusters of methods: **word overlap-based metrics** and **embedding-based metrics**. 

Word overlap-based metrics focus on evaluating word overlap between predicted sentences and target sentences (Jaccard similarity, word error rate, BLUE score). While these methods are useful for ensuring that predictions are lexically similar, they fail to capture semantically similar sentences that do not necessarily of have common words. In addition, two sentences may have a high number of common words but very different meanings overall.   

Embedding-based metrics consider meanings of sentences by combining word embeddings (Word2Vec, Glove, etc) and compute a distance measure (cosine distance, Word Mover’s Distance) between embedding vectors of predicted and target sentences. These methods take into account words’ similarities in a word embedding space allowing to capture semantic similarities irrespective of common words. 

Here, metrics from both categories are implemented. 

## How To Use

Create a conda environment and install required packages
```
conda env create -f env.yml
conda activate paraphraser_env
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
```
python3 evaluate.py --test_path test_data.csv --device cpu
```

## References

- Graham Neubig, Neural Machine Translation and Sequence-to-sequence Models: A Tutorial. Carnegie Mellon University (2017). [Graham Neubig](https://arxiv.org/pdf/1703.01619.pdf)

- Stanford CS224n Natural Language Processing with Deep Learning [CS224n](http://web.stanford.edu/class/cs224n/)

- Adrien Sieg, Text Similarities: Estimate the degree of similarity between two texts. [Adrien Sieg] (https://medium.com/@adriensieg/text-similarities-da019229c894)

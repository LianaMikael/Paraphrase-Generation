from jiwer import wer 
import nltk 
import numpy as np 
import scipy

class Evaluator:
    ''' Evaluates a batch of predicted and target sentences 
        Supports: average WER, corpus BLEU, GloVe + cosine '''

    def __init__(self, predictions, targets, WER=None, BLEU=None, glove_file=None):
        self.check_input(predictions, targets)

        self.clean_targets = []
        self.clean_predictions = []
        for pred, target in zip(predictions, targets):
            if len(target) > 0 and len(pred) > 0: 
                self.clean_targets.append(target)
                self.clean_predictions.append(pred)
        
        if WER:
            self.WER = self.word_error_rate()
        if BLEU:
            self.BLEU = self.BLEU_score()
        if glove_file:
            embed_dict = self.load_glove(glove_file)
            self.embed_sim = self.embed_similarity(embed_dict)

    def BLEU_score(self):
        return nltk.translate.bleu_score.corpus_bleu(self.clean_targets, self.clean_predictions, weights=(1, 0, 0, 0))

    def word_error_rate(self):
        wers = []
        for pred, target in zip(self.clean_predictions, self.clean_targets):
            wers.append(wer(target, pred))
        return np.mean(wers)

    def load_glove(self, glove_file):
        print('Loading GloVe embeddings...')
        embed_dict = {}
        with open(glove_file, 'r+') as f:
            for line in f:
                word = line.split()[0]
                embedding = np.array([float(val) for val in line.split()[1:]])
                embed_dict[word] = embedding
        
        return embed_dict

    def embed_similarity(self, embed_dict):
        # constructs sentence embeddings by retrieving GloVe embeddings for each word in each sentence
        # calculates cosine similarity between a prediction and a target sentence embeddings 
        similarities = []
        for i in range(len(self.clean_predictions)):
            pred = self.remove_stop_words(self.clean_predictions[i])
            target = self.remove_stop_words(self.clean_targets[i])

            pred_embed = np.mean([embed_dict.get(word, 0) for word in pred])
            target_embed = np.mean([embed_dict.get(word, 0) for word in target]) 

            cos_sim = scipy.spatial.distance.cosine(pred_embed, target_embed)
            print(cos_sim)
            similarities.append(cos_sim)
        
        return np.mean(similarities)

    @staticmethod
    def remove_stop_words(sentence):
        clean_words = []
        for word in sentence:
            if word not in nltk.corpus.stopwords.words("english"):
                clean_words.append(word)
        return clean_words

    @staticmethod
    def check_input(predictions, targets):
        assert len(predictions) == len(targets)
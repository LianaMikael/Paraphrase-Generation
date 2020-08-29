from absl import app, logging, flags
import json
import nltk
from nltk.tokenize import RegexpTokenizer
import sys

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'train_data.csv.', 'Data file path (csv)')
flags.DEFINE_string('vocab_path', 'vocab.json', 'Vocabulary file path (json)')
flags.mark_flag_as_required('data_path')

class Vocabulary():
    ''' constructs vocabulary for source and target sentences '''
    def __init__(self, source_vocab, target_vocab):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab 

        self.source_id_word = {word: i for word, i in self.source_vocab.items()}
        self.target_id_word = {word: i for word, i in self.target_vocab.items()}

    @staticmethod
    def construct_vocab_entry(corpus):
        # initializes word -> index dictionary inclduing pad, start, end and uknown tokens 
        word_index = {}

        word_index['<pad>'] = 0
        word_index['<s>'] = 1
        word_index['</s>'] = 2
        word_index['<unk>'] = 3

        for sentence in corpus:
            for word in sentence:
                if word not in word_index:
                    word_index[word] = len(word_index)

        return word_index

    @staticmethod
    def build(source_sentences, target_sentences):

        assert len(source_sentences) == len(target_sentences)

        source_vocab = Vocabulary.construct_vocab_entry(source_sentences)
        target_vocab = Vocabulary.construct_vocab_entry(target_sentences)

        return Vocabulary(source_vocab, target_vocab)

    def save(self, vocab_path):
        vocab = {'source_data': self.source_vocab, 'target_data': self.target_vocab}
        f_out = open(vocab_path, 'w')
        json.dump(vocab, f_out, indent = 2)

        print('Vocabulary saved into ' + vocab_path)

    @staticmethod
    def load(vocab_path):
        with open(vocab_path, 'r+') as f:
            word_index = json.load(f)
            source_vocab = word_index['source_data']
            target_vocab = word_index['target_data'] 

        return Vocabulary(source_vocab, target_vocab)

def read(data_path):

    tokenizer = RegexpTokenizer(r'\w+')

    source_sentences = []
    target_sentences = []
    with open(data_path, 'r+') as f:
        for line in f:
            line = line.split(',')
            source_tokens = tokenizer.tokenize(line[0].lower())
            target_tokens = tokenizer.tokenize(line[1].lower())

            source_sentences.append(source_tokens)
            target_sentences.append(['<s>'] + target_tokens + ['</s>']) # adds the start and end tokens to the target sentences 

    return source_sentences, target_sentences

def main(_):
    # reads the data file containing both the source and the target sentences, builds the vocabulary and saves into a json file 

    data_path = FLAGS.data_path
    vocab_path = FLAGS.vocab_path

    source_sentences, target_sentences = read(data_path)

    vocab = Vocabulary.build(source_sentences, target_sentences)
    vocab.save(vocab_path)

if __name__ == '__main__':
    app.run(main)



    




        

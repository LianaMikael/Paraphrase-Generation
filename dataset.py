import torch 
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    ''' Dataset for constructing source and target tensors '''

    def __init__(self, source_sentences, target_sentences, vocab, num_tokens=50):
        # source_sentences: list of lists of tokens
        # target_sentences: list of lists of tokens 

        assert len(source_sentences) == len(target_sentences)
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.num_tokens = num_tokens
        self.vocab = vocab

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source = self.convert_to_tensor(self.source_sentences[idx], self.vocab.source_vocab)
        target = self.convert_to_tensor(self.target_sentences[idx], self.vocab.target_vocab)
        return source, target

    def convert_to_tensor(self, sentence, tokens_vocab):
        # converts a list of tokens into padded tokens using the corresponding vocabulary 
        # for out of vocabulary words, assigns '<unk>'

        word_ids = [tokens_vocab.get(word, tokens_vocab['<unk>']) for word in sentence]
        padded_ids = self.pad_sentences(word_ids)
        return torch.tensor(padded_ids, dtype=torch.long)

    def pad_sentences(self, sentence):
        # pads a given single sentence
        assert len(sentence) > 0
        if len(sentence) < self.num_tokens:
            return sentence + [0] * (self.num_tokens - len(sentence))
        return sentence[:self.num_tokens]
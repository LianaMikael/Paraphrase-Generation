import torch 
from torch.utils.data import Dataset, DataLoader

class SentenceDataset(Dataset):
    ''' Dataset for constructing batches of source and target tensors '''

    def __init__(self, source_sentences, target_sentences, vocab):
        # source: list of lists of tokens
        # target: list of lists of tokens 

        self.source = self.convert_to_tensor(source_sentences, vocab.source_vocab)
        self.target = self.convert_to_tensor(target_sentences, vocab.target_vocab)
        assert self.source.shape[0] == self.target.shape[0]

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

    def convert_to_tensor(self, tokens_list, tokens_vocab):
        # converts list of lists of tokens into padded tokens using the corresponding vocabulary 
        # for out of vocabulary words, assigns '<unk>'

        word_ids = [[tokens_vocab.get(word, tokens_vocab['<unk>']) for word in s] for s in tokens_list]
        padded_ids = self.pad_sentences(word_ids)
        return torch.tensor(padded_ids, dtype=torch.long)

    @staticmethod
    def pad_sentences(sentences):
        # pads given list of senteces
        
        lengths = [len(s) for s in sentences]
        padded_sents = []
        for i in range(len(sentences)):
            padded_sents.append(sentences[i] + [0] * (max(lengths) - lengths[i]))

        return padded_sents
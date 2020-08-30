import sys
import torch
import torch.nn as nn 
import torch.nn.functional as F

from collections import namedtuple

class Embeddings(nn.Module):
    ''' converts source and target sentences into their embeddings '''

    def __init__(self, embed_size, vocabulary):
        super(Embeddings, self).__init__()

        self.embed_size = embed_size 

        self.source = nn.Embedding(len(vocabulary.source_vocab), embed_size, vocabulary.source_vocab['<pad>'])
        self.target = nn.Embedding(len(vocabulary.target_vocab), embed_size, vocabulary.target_vocab['<pad>'])


class Paraphraser(nn.Module):
    ''' sequence-to-sequnce model 
        bidirectional LSTM encoder
        unidirectional LSTM decoder
        global attention model ''' 

    def __init__(self, embed_size, hidden_size, vocabulary, device, dropout_rate=0.3):
        super(Paraphraser, self).__init__()

        self.embeddings = Embeddings(embed_size, vocabulary)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate 
        self.embed_size = embed_size
        self.vocab = vocabulary

        self.device = self.embeddings.source.weight.device

        # initialize endocer and decoder LSTMs
        self.encoder = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, bidirectional = True)
        self.decoder = nn.LSTMCell(input_size = self.embed_size + self.hidden_size, hidden_size = self.hidden_size)

        # initialize linear projections of encoder's final hidden state and cell state
        self.h_proj = nn.Linear(in_features = 2*self.hidden_size, out_features = self.hidden_size, bias = False)
        self.c_proj = nn.Linear(in_features = 2*self.hidden_size, out_features = self.hidden_size, bias = False)

        # projection layer for multiplicative attention 
        self.attention_proj = nn.Linear(in_features = 2*self.hidden_size, out_features = self.hidden_size, bias = False)

        # combined output projection layer 
        self.combined_output_proj = nn.Linear(in_features = 3*self.hidden_size, out_features = self.hidden_size, bias=False)

        # target words projection layer to be able to produce a probability distribution 
        self.target_vocab_proj = nn.Linear(in_features = self.hidden_size, out_features = len(self.vocab.target_vocab), bias = False)

        # dropout rate for attention 
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, source, target):
        # computes a probability of composed target senteces for a given batch 
        # source: list of source sentence tokens
        # target: list of target sentence tokens including start and end tokens

        source_lengths = [len(s) for s in source]

        # convert list of lists of tokens into padded tokens
        # for out of vocabulary words, assign '<unk>'
        word_ids = [[self.vocab.source_vocab.get(word, self.vocab.source_vocab['<unk>']) for word in s] for s in source]
        padded_source = self.pad_sentences(word_ids)

        word_ids_target = [[self.vocab.target_vocab.get(word, self.vocab.target_vocab['<unk>']) for word in s] for s in target]
        padded_target = self.pad_sentences(word_ids_target)

        # convert padded source and target lists into tensors 
        padded_source = torch.t(torch.tensor(padded_source, dtype=torch.long, device=self.device))
        padded_target = torch.t(torch.tensor(padded_target, dtype=torch.long, device=self.device))

        # apply encoder function to padded source sentences to obtain the encoder hidden state and the decoder initil state
        enc_hidden, dec_init_state = self.encode(padded_source, source_lengths)

        # genrate encoder masks 
        enc_masks = self.generate_masks(enc_hidden, source_lengths)

        # compute combined output vector constructed by the decode function 
        combined_vec = self.decode(enc_hidden, enc_masks, dec_init_state, padded_target)

        # compute the log-probability of distribution over target words 
        probs = F.log_softmax(self.target_vocab_proj(combined_vec), dim=-1)

        # zero out the probability for the padding tokens since there are not present in the target corpus 
        padded_target = padded_target[1:] # to remove the start token 
        target_masks = (padded_target != self.vocab.target_vocab['<pad>']).float()

        # compute the log-probability distribution over the true target words 
        indices = padded_target.unsqueeze(-1)
      
        final_words_probs = torch.gather(probs, index=indices, dim=-1).squeeze(-1) * target_masks
        final_scores = torch.sum(final_words_probs, dim=0)

        return final_scores

    def encode(self, source_padded, source_lengths):
        # using the embeddings, contrust a tensor of source sentences (source_len, batch_size, embed_size)
        source_embed = self.embeddings.source(source_padded)
    
        # pad the packed sequence source_embed for fewer computations
        source_embed = nn.utils.rnn.pack_padded_sequence(source_embed, source_lengths)

        # apply the encoder bidirectional LSTM to obatin the encoder hidden state, final hidden (forward and backward) state and the cell state 
        enc_hidden, (final_hidden, final_cell) = self.encoder(source_embed)

        # concatenate the forward and backward hidden states 
        final_hidden_state = torch.cat([final_hidden[0], final_hidden[1]], dim=1)

        # concatenate the forward and backward cell states 
        final_cell_state = torch.cat([final_cell[0], final_cell[1]], dim=1)

        # pad back the encoder hidden states
        enc_hidden = nn.utils.rnn.pad_packed_sequence(enc_hidden)[0]

        # permute the tensor from (source_len, batch_size, 2*hidden_size) to (batch_size, source_len, 2*hidden_size)
        enc_hidden = enc_hidden.permute(1, 0, 2)

        # construct the initial state of the decoder by applying the linear projections
        final_hidden_proj = self.h_proj(final_hidden_state)
        final_cell_proj = self.c_proj(final_cell_state)

        return enc_hidden, (final_hidden_proj, final_cell_proj)

    def decode(self, enc_hidden, enc_masks, dec_init, padded_target):

        # initialize the decoder state 
        dec_state = dec_init 

        # initialize the previous combined output vector as zeros 
        o_prev = torch.zeros(enc_hidden.shape[0], self.hidden_size, device=self.device)

        # apply attention projection layer to encoder hidden states
        enc_hidden_proj = self.attention_proj(enc_hidden)

        # get the target embeddings (without the end token)
        target_embed = self.embeddings.target(padded_target[:-1])

        combined_outputs = []

        # for every time step of the target embedding
        # concatenate embeddings with the previous combined vector output
        # compute decoder's next step 
        for embed_t in target_embed:
            new_embed = torch.cat([embed_t, o_prev], dim=1)

            dec_state, o_t = self.step(new_embed, dec_state, enc_hidden, enc_hidden_proj, enc_masks)

            combined_outputs.append(o_t)
            o_prev = o_t
            
        return torch.stack(combined_outputs)


    def generate_masks(self, enc_hidden, source_lengths):
        # generate encoder masks to zero out the padded tokens 
        enc_masks = torch.zeros(enc_hidden.shape[0], enc_hidden.shape[1], dtype=torch.float)

        count = 0
        for i in range(len(source_lengths)):
            enc_masks[count, source_lengths[0]:] = 1
            count += 1

        return enc_masks

    def step(self, new_embed, dec_state, enc_hidden, enc_hidden_proj, enc_masks):
        # one forward step of the decoder 

        # apply decoder LSTM to the embeddings and the previous state to obtain the updated decoder state
        dec_state = self.decoder(new_embed, dec_state)
        dec_hidden, dec_cell = dec_state 

        # compute attention scores 
        att_scores = torch.bmm(enc_hidden_proj, dec_hidden.unsqueeze(2)).squeeze(2)

        # set -inf, attention socres for padding tokens
        if enc_masks is not None:
            att_scores.masked_fill_(enc_masks.to(torch.uint8), -float('inf'))

        # apply softmax to attention scores to obtain attention probability distribution 
        att = F.softmax(att_scores, dim=1)

        # attention output vector 
        final_att = torch.bmm(att.unsqueeze(1), enc_hidden).squeeze(1)

        # concatenate attenton output vector with decoder hidden states 
        dec_combined = torch.cat([dec_hidden, final_att], dim=1)

        # pass it through a linear layer, tanh and dropout 
        o_t = self.dropout(torch.tanh(self.combined_output_proj(dec_combined)))

        return dec_state, o_t

    @staticmethod
    def pad_sentences(sentences):
        # pads given list of senteces with the pad token '<pad>'
        lengths = [len(s) for s in sentences]

        padded_sents = []
        for i in range(len(sentences)):
            padded_sents.append(sentences[i] + [0] * (max(lengths) - lengths[i]))

        return padded_sents

    @staticmethod
    def load(path):
        # loads a saved model 
        params = torch.load(path)
        model = Paraphraser(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])

        print('Loaded the model from ' + path)

        return model 

    def save(self, path):
        # saves the model 

        params = {
            'args': dict(embed_size=self.embeddings.embed_size, hidden_size = self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

        print('Saved model to ' + path)





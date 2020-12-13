from absl import app, logging, flags
import torch
import torch.nn.functional as F
import numpy as np 
from model import Paraphraser
from construct_vocabulary import read
from train import train

from jiwer import wer 
import nltk 
from collections import namedtuple

FLAGS = flags.FLAGS

flags.DEFINE_string('test_path', 'val_data.csv', 'Test file path (csv)')
flags.DEFINE_string('save_output', 'output.csv', 'Path to save output')
flags.DEFINE_integer('beam_size', 10, 'Beam size')
flags.DEFINE_integer('max_decode_time', 25, 'Maximum decoding time step')

flags.mark_flag_as_required('load_model')

def decode(test_path, load_model, beam_size, max_t, hidden_size, output_file, device, show_each=True):
    # performs decoding 
    test_source, test_target = read(test_path)

    print('Loading model from {}'.format(load_model))
    model = Paraphraser.load(load_model, device)
    model.to(device)

    all_greedy_sents = []
    all_beam_search_sents = []
    all_test_target = []

    f = open(output_file, 'w+')
    f.write('{},{},{},{}\n'.format('source sentence', 'target sentence', 'greedy hypothesis', 'beam search hypothesis'))

    # perform beam search and greedy decoding for each source sentence 
    for i in range(len(test_source)):
        source_sentence = test_source[i]
        target_sentence = test_target[i]

        if len(source_sentence) > 0:
            all_test_target.append(target_sentence)

            greedy_hypothesis, all_hypotheses = get_hypothesis(source_sentence, model, beam_size, max_t, hidden_size, device)

            all_greedy_sents.append(greedy_hypothesis)
            all_beam_search_sents.append(all_hypotheses[0][0])

            if show_each:
                print('source:', ' '.join(source_sentence))
                print('greedy:', ' '.join(greedy_hypothesis))
                print('beam search:', ' '.join(all_hypotheses[0][0]))
                print('target:', ' '.join(target_sentence[1:-1]))
                print()

            f.write('{},{},{},{}\n'.format(' '.join(source_sentence), ' '.join(target_sentence[1:-1]), ' '.join(greedy_hypothesis), ' '.join(all_hypotheses[0][0])))
    
    print('Results saved into ', output_file)
    f.close()
        
    return all_test_target, all_greedy_sents, all_beam_search_sents


def get_hypothesis(source_sentence, model, beam_size, max_t, hidden_size, device):
    # performs beam search and greedy decoding for a single sentence to estimate best decoding sequences 

    Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

    # convert source sentence to tensor of source word ids 
    source_words = [model.vocab.source_vocab.get(word, model.vocab.source_vocab['<unk>']) for word in source_sentence]
    source_tensor = torch.t(torch.tensor(source_words, dtype=torch.long, device=device))

    source_tensor = source_tensor.unsqueeze(1) 
    
    # apply encoder and decoder functions to the source sentence 
    enc_hidden, dec_init_state = model.encode(source_tensor, [len(source_sentence)])

    dec_state_h = dec_init_state 

    att = torch.zeros(1, hidden_size, device=device)

    enc_hidden_att = model.attention_proj(enc_hidden)

    target_id_word = {i : word for word, i in model.vocab.target_vocab.items()}

    # collect hypotheses and their scores 
    hypotheses = [['<s>']]
    scores = torch.zeros(len(hypotheses), dtype=torch.float, device = device)
    all_hypotheses = []

    greedy_hypothesis = []
    greedy_end = False

    # loop through each time step 
    for _ in range(max_t):

        enc_hidden_new = enc_hidden.expand(len(hypotheses), enc_hidden.size(1), enc_hidden.size(2))
        enc_hidden_att_new = enc_hidden_att.expand(len(hypotheses), enc_hidden_att.size(1), enc_hidden_att.size(2))  

        target_tensor = torch.tensor([model.vocab.target_vocab[h[-1]] for h in hypotheses], dtype=torch.long, device=device)
        target_embed = model.embeddings.target(target_tensor)

        new_embed = torch.cat([target_embed, att], dim=-1)

        # only perform one decoder step 
        dec_state, new_att, _ = model.step(new_embed=new_embed, dec_state=dec_state_h, enc_hidden=enc_hidden_new, enc_hidden_proj=enc_hidden_att_new, enc_masks=None)

        probs = F.log_softmax(model.target_vocab_proj(new_att), dim=-1)

        h_scores = (scores.unsqueeze(1).expand_as(probs) + probs).view(-1)
        
        if not greedy_end:
            # perform greedy decoding by selecting the word with the highest score at each step 
            h_scores_exp = torch.exp(h_scores)
            greedy_id = torch.argmax(h_scores_exp) % len(model.vocab.target_vocab)
            greedy_word = target_id_word.get(greedy_id.item(), '<unk>')

            if greedy_word != '</s>': 
                greedy_hypothesis.append(greedy_word)
            else:
                # stop greedy decoding when end token is reached 
                greedy_end = True


        # select top k scores for beam decoding 
        top_h_scores, top_h_pos = torch.topk(h_scores, k=beam_size - len(all_hypotheses))

        prev_h_ids = top_h_pos // len(model.vocab.target_vocab)
        h_word_ids = top_h_pos % len(model.vocab.target_vocab)

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_h_id, h_word_id, new_h_score in zip(prev_h_ids, h_word_ids, top_h_scores):
            prev_h_id = prev_h_id.item()
            h_word_id = h_word_id.item()
            new_h_score = new_h_score.item()

            potential_word = target_id_word.get(h_word_id, '<unk>')
           
            new_hyp_sent = hypotheses[prev_h_id] + [potential_word]
            
            # save the hypothesis sentence when it reaches the end token 
            if potential_word == '</s>':
                all_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                        score=new_h_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_h_id)
                new_hyp_scores.append(new_h_score)

        # stop when maximum number of hypotheses reached 
        if len(all_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)

        dec_state_h = (dec_state[0][live_hyp_ids], dec_state[1][live_hyp_ids])
        att = new_att[live_hyp_ids]

        hypotheses = new_hypotheses
        scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

    if len(all_hypotheses) == 0:
        all_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=scores[0].item()))

    all_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
    
    return greedy_hypothesis, all_hypotheses

def evaluate_word_level(targets, predictions):
    # evaluate results with corpus-level BLEU score, average word error rate and Jaccard similarity 

    assert len(predictions) == len(targets)

    wers = []
    clean_targets = []
    clean_predictions = []
    for pred, target in zip(predictions, targets):
        if len(target) > 0 and len(pred) > 0: 
            wers.append(wer(target, pred))
            clean_targets.append(target)
            clean_predictions.append(pred)
            
    BLEU = nltk.translate.bleu_score.corpus_bleu(clean_targets, clean_predictions)

    return np.mean(wers), BLEU

def main(_):
    test_path = FLAGS.test_path
    device = FLAGS.device

    load_model = FLAGS.load_model
    beam_size = FLAGS.beam_size
    max_t = FLAGS.max_decode_time

    output_file = FLAGS.save_output
    hidden_size = FLAGS.hidden_size

    device = torch.device('cuda:0' if FLAGS.device=='cuda' else 'cpu')

    print('Started decoding...')
    targets, greedy_sents, beam_search_sents = decode(test_path, load_model, beam_size, max_t, hidden_size, output_file, device)

    print('Decoding completed.')

    wer_score_bs, BLEU_score_bs = evaluate_word_level(targets, beam_search_sents)
    wer_score_greey, BLEU_score_greedy = evaluate_word_level(targets, greedy_sents)

    print('Word Error Rate with beam search decoding: {}, with greedy decoding {}'.format(wer_score_bs, wer_score_greey))
    print('Corpus level BLEU score with beam search decoding: {}, with greeedy decoding: {} '.format(BLEU_score_bs, BLEU_score_greedy))

if __name__ == '__main__':
    app.run(main)


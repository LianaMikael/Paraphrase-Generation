from absl import app, logging, flags
import torch 
import torch.nn as nn
import numpy as np 
from model import Paraphraser
from construct_vocabulary import Vocabulary, read

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', 'train_data.csv', 'Train file path (csv)')
flags.DEFINE_string('val_path', 'val_data.csv', 'Validation file path (csv)')
flags.DEFINE_integer('train_batch_size', 256, 'Train batch size')
flags.DEFINE_integer('val_batch_size', 128, 'Validation batch size')
flags.DEFINE_string('save_model', 'test.bin', 'Model save path')
flags.DEFINE_string('load_model', None, 'Model load path')
flags.DEFINE_string('device', 'cuda', 'Device')

flags.DEFINE_integer('embed_size', 512, 'embeddings dimentionality')
flags.DEFINE_integer('hidden_size', 512, 'LSTM hidden size')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 5, 'Max number of epochs')
flags.DEFINE_integer('save_every', 200, 'Evaluate and save iterations')
flags.DEFINE_integer('display_every', 10, 'Display training details')

def train(model, vocab, train_path, val_path, train_batch_size, val_batch_size, embed_size, hidden_size, lr, epochs, save_model, save_every, display_every, device):
    # performs model training 
    train_data_source, train_data_target = read(train_path)
    val_data_source, val_data_target = read(val_path)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr)

    examples = list(zip(train_data_source, train_data_target))

    # for each epoch, perform training on a shuffled mini-batch of sentences 
    total_loss = 0
    report_loss = 0
    report_examples = 0
    train_i = 0 
    val_perplexities = []

    for epoch in range(0, epochs):
        np.random.shuffle(examples)
        batches = generate_batches(examples, train_batch_size)
        for batch in batches:

            train_i += 1
            source_batch, target_batch = batch

            padded_source, padded_target, source_lengths = convert_tensors(source_batch, target_batch, vocab, device)
 
            optimizer.zero_grad()

            current_batch_size = len(source_batch)
            current_losses = - model(padded_source, padded_target, source_lengths)
            batch_loss = current_losses.sum()
            loss = batch_loss / current_batch_size 
            
            loss.backward()
            
            # clip gradient to avoid gradient explosion 
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            report_loss += batch_loss.item()
            total_loss += batch_loss.item()
            report_examples += train_batch_size 

            # report training statistics  
            if train_i % display_every == 0:
                print('epoch {}, train iter {}, average loss {}'.format(epoch+1, train_i, report_loss/report_examples))
                report_loss = 0
                report_examples = 0

            # evaluate and save the model with the best val loss
            if train_i % save_every == 0:
                print('epoch {}, train iter {}, val loss {}'.format(epoch+1, train_i, perplexity))
                print()
        
                perplexity = validate(model, val_data_source, val_data_target, val_batch_size, vocab, device)

                if len(val_perplexities) == 1 or perplexity <= min(val_perplexities):
                    model.save(save_model)

                model.train()
                
def validate(model, val_data_source, val_data_target, val_batch_size, vocab, device):
    # performs validation by calculating perplexity of a batch given the current model 
    model.eval()
    total_val_loss = 0
    total_val_words = 0

    with torch.no_grad():
        val_data = generate_batches(list(zip(val_data_source, val_data_target)), val_batch_size)
        
        for val_batch in val_data:
            val_source_batch, val_target_batch = val_batch

            padded_source, padded_target, source_lengths = convert_tensors(val_source_batch, val_target_batch, vocab, device)

            val_losses = - model(padded_source, padded_target, source_lengths)
            val_batch_loss = val_losses.sum()
            
            total_val_loss += val_batch_loss.item()
            total_val_words += val_batch_size 
        
        perplexity = total_val_loss / total_val_words

    return perplexity

def convert_tensors(source, target, vocab, device):
    # converts list of lists of tokens into padded tokens

    source_lengths = [len(s) for s in source]
    # for out of vocabulary words, assign '<unk>'
    word_ids = [[vocab.source_vocab.get(word, vocab.source_vocab['<unk>']) for word in s] for s in source]
    padded_source = pad_sentences(word_ids)

    word_ids_target = [[vocab.target_vocab.get(word, vocab.target_vocab['<unk>']) for word in s] for s in target]
    padded_target = pad_sentences(word_ids_target)

    padded_source = torch.t(torch.tensor(padded_source, dtype=torch.long, device=device))
    padded_target = torch.t(torch.tensor(padded_target, dtype=torch.long, device=device))

    return padded_source, padded_target, source_lengths

def pad_sentences(sentences):
    # pads given list of senteces
    lengths = [len(s) for s in sentences]

    padded_sents = []
    for i in range(len(sentences)):
        padded_sents.append(sentences[i] + [0] * (max(lengths) - lengths[i]))

    return padded_sents

def generate_batches(examples, batch_size):
    # generates mini-batches of source and target senteces for training in decreasing order
    
    full_batches =[]
    for i in range(int(np.ceil(len(examples)/batch_size))):

        batch = []
        for j in range(i * batch_size, (i+1) * batch_size):
            
            if j < len(examples) and len(examples[j][0]) > 0 and len(examples[j][1]) > 0: 
                batch.append(examples[j])

        sorted_examples = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        
        source_batch = [x[0] for x in sorted_examples]
        target_batch = [x[1] for x in sorted_examples]
        
        full_batches.append([source_batch, target_batch])
    
    return full_batches

def main(_):

    train_path = FLAGS.train_path
    val_path = FLAGS.val_path
    vocab_path = FLAGS.vocab_path
    train_batch_size = FLAGS.train_batch_size
    val_batch_size = FLAGS.val_batch_size
    save_model = FLAGS.save_model

    embed_size = FLAGS.embed_size
    hidden_size = FLAGS.hidden_size
    lr = FLAGS.lr
    epochs = FLAGS.epochs

    save_every = FLAGS.save_every
    display_every = FLAGS.display_every 

    device = torch.device('cuda:0' if FLAGS.device=='cuda' else 'cpu')
    vocab = Vocabulary.load(vocab_path)

    if FLAGS.load_model:
        load_model = FLAGS.load_model
        model = Paraphraser.load(load_model, device)

    else:
        model = Paraphraser(embed_size, hidden_size, vocab, device)
         # uniformly initialize the parameters
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

    print('Started training... ')
    train(model, vocab, train_path, val_path, train_batch_size, val_batch_size, embed_size, hidden_size, lr, epochs, save_model, save_every, display_every, device)

if __name__ == '__main__':
    app.run(main)


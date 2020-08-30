from absl import app, logging, flags
import torch 
import torch.nn as nn
import numpy as np 
from model import Paraphraser
from construct_vocabulary import Vocabulary, read
#print(torch.__version__)

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', 'train_data.csv', 'Train file path (csv)')
flags.DEFINE_string('val_path', 'val_data.csv', 'Validation file path (csv)')
#flags.DEFINE_string('vocab_path', 'vocab.json', 'Vocabulary file path (json)')
flags.DEFINE_integer('train_batch_size', 32, 'Train batch size')
flags.DEFINE_integer('val_batch_size', 64, 'Validation batch size')
flags.DEFINE_string('save_model', 'model.bin', 'Model save path')
flags.DEFINE_string('device', 'cuda', 'Device')

flags.DEFINE_integer('embed_size', 512, 'embeddings dimentionality')
flags.DEFINE_integer('hidden_size', 512, 'LSTM hidden size')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 5, 'Max number of epochs')

#flags.mark_flag_as_required('data_path')
#flags.mark_flag_as_required('val_path')


def train(train_path, val_path, vocab_path, train_batch_size, val_batch_size, embed_size, hidden_size, lr, epochs, save_model, device):
    # performs model training 
    train_data_source, train_data_target = read(train_path)
    val_data_source, val_data_target = read(val_path)

    vocab = Vocabulary.load(vocab_path)

    model = Paraphraser(embed_size, hidden_size, vocab, device)
    model.train()

    # uniformly initialize the parameters
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    examples = list(zip(train_data_source, train_data_target))

    # for each epoch, perform training on a shuffled mini-batch of sentences 
    total_loss = 0
    report_loss = 0
    report_examples = 0
    train_i = 0
    for epoch in range(0,epochs):
        np.random.shuffle(examples)
        batches = generate_batches(examples, train_batch_size)
        for batch in batches:

            train_i += 1
            source_batch, target_batch = batch

            optimizer.zero_grad()

            current_batch_size = len(source_batch)

            current_losses = - model(source = source_batch, target = target_batch)
            batch_loss = current_losses.sum()
            loss = batch_loss / current_batch_size 
            
            loss.backward()
            
            # clip gradient to avoid gradient explosion 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            report_loss += batch_loss.item()
            total_loss += batch_loss.item()
            report_examples += train_batch_size 

            # report results every 1000 iterations 
            if train_i % 1000 == 0:
                print('epoch {}, train iter {}, average loss {}'.format(epoch+1, train_i, report_loss/report_examples))
                report_loss = 0
                report_examples = 0

                model.eval()
                total_val_loss = 0
                total_val_words = 0
                # perform validation every 1000 iterations
                with torch.no_grad():
                    val_data = generate_batches(list(zip(val_data_source, val_data_target)), 32)
                    
                    val_losses = - model(source = val_data[0][0], target = val_data[0][1])
                    val_batch_loss = val_losses.sum()
                    
                    total_loss += val_batch_loss.item()
                    total_val_words += sum(len(s[1:]) for s in val_data_target)
                    perplexity = np.exp(total_loss / total_val_words)

                print('epoch {}, train iter {}, perplexity {}'.format(epoch+1, train_i, perplexity))
                model.train()
    # save the model after every epoch 
    model.save(save_model)

def generate_batches(examples, batch_size):
    # generates mini-batches of source and target senteces for training in decreasing order
    
    full_batches =[]
    for i in range(int(np.ceil(len(examples)/batch_size))):

        batch = []
        for i in range(batch_size):
            if len(examples[i][0]) > 0 and len(examples[i][1]) > 0: 
                batch.append(examples[i])

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

    device = torch.device('cuda:0' if FLAGS.device=='cuda' else 'cpu')

    embed_size = FLAGS.embed_size
    hidden_size = FLAGS.hidden_size
    lr = FLAGS.lr
    epochs = FLAGS.epochs

    train(train_path, val_path, vocab_path, train_batch_size, val_batch_size, embed_size, hidden_size, lr, epochs, save_model, device)


if __name__ == '__main__':
    app.run(main)


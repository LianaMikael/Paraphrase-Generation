from absl import app, logging, flags
import torch 
import torch.nn as nn
import numpy as np 
from model import Paraphraser
from dataset import SentenceDataset
from construct_vocabulary import Vocabulary, read
import torch.onnx 
import onnx

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', 'train_data_all.csv', 'Train file path (csv)')
flags.DEFINE_string('val_path', 'val_data_all.csv', 'Validation file path (csv)')
flags.DEFINE_integer('train_batch_size', 256, 'Train batch size')
flags.DEFINE_integer('val_batch_size', 128, 'Validation batch size')
flags.DEFINE_string('save_model', 'model.bin', 'Model save path')
flags.DEFINE_string('load_model', None, 'Model load path')
flags.DEFINE_string('device', 'cuda', 'Device')

flags.DEFINE_integer('embed_size', 512, 'embeddings dimentionality')
flags.DEFINE_integer('hidden_size', 512, 'LSTM hidden size')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 5, 'Max number of epochs')
flags.DEFINE_integer('save_every', 100, 'Evaluate and save iterations')
flags.DEFINE_integer('display_every', 10, 'Display training details')

def train(model, vocab, train_loader, val_loader, optimizer, embed_size, hidden_size, epochs, save_model, save_every, display_every, device):
    model.train()
    
    total_loss = 0
    report_loss = 0
    report_examples = 0
    total_target_words = 0
    train_i = 0 
    val_perplexities = []

    for epoch in range(0, epochs):
        for batch in train_loader:

            train_i += 1
            
            padded_source, padded_target = process_batch(batch)

            optimizer.zero_grad()

            current_losses = - model(padded_source, padded_target)
            batch_loss = current_losses.sum()
            loss = batch_loss / padded_source.shape[0]
            
            loss.backward()
            
            # clip gradient to avoid gradient explosion 
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            report_loss += batch_loss.item()
            total_loss += batch_loss.item()
            report_examples += padded_source.shape[0]  

            # calculate perplexity by averaging the loss over all words 
            total_target_words += sum([len(padded_target[i][padded_target[i] != 0])-1 for i in range(padded_target.shape[0])])
            train_perplexity = np.exp(report_loss/total_target_words)

            # report training statistics  
            if train_i % display_every == 0:
                print('epoch {}, train iter {}, average train loss {}, train perplexity {}'.format(epoch+1, train_i, report_loss/report_examples, train_perplexity))
                report_loss = 0
                report_examples = 0

            # evaluate and save the model with the best val loss
            if train_i % save_every == 0:
        
                val_loss, val_perplexity = validate(model, val_loader, vocab, device)
                val_perplexities.append(val_perplexity)
                print('epoch {}, train iter {}, average val loss {}, val perplexity {}'.format(epoch+1, train_i, val_loss, val_perplexity))
                print()

                if len(val_perplexities) == 1 or val_perplexity <= min(val_perplexities):
                    model.save(save_model)

                model.train()

def validate(model, val_loader, vocab, device):
    # performs validation by calculating perplexity of a batch given the current model 
    model.eval()
    total_val_loss = 0
    total_val_words = 0
    total_target_words = 0

    with torch.no_grad():
        
        for val_batch in val_loader:
            padded_source, padded_target = process_batch(val_batch)

            val_losses = - model(padded_source, padded_target)
            val_batch_loss = val_losses.sum()
            
            total_val_loss += val_batch_loss.item()
            total_val_words += padded_source.shape[0] 

            total_target_words += sum([len(padded_target[i][padded_target[i] != 0])-1 for i in range(padded_target.shape[0])])

        loss = total_val_loss / total_val_words
        perplexity = np.exp(total_val_loss / total_target_words)
    
    return loss, perplexity

def process_batch(batch):
    # sort batch according to the lengths of sentences in the source 

    padded_source, padded_target = batch
    examples_batch = torch.cat((padded_source, padded_target), dim=1)
    sorted_examples = sorted(examples_batch, key=lambda x: len(x[:padded_source.shape[1]+1][x[:padded_source.shape[1]+1]!=0]), reverse=True)
    sorted_examples = torch.stack(sorted_examples, dim=0)
   
    padded_source = sorted_examples[:,:padded_source.shape[1]]
    padded_target = sorted_examples[:,padded_source.shape[1]:]

    return padded_source, padded_target

def convert_onnx(model, val_loader, vocab):
    # exports the pytorch model to onnx 
    # accepts source and target tensor with batch_size and padded_length of variable sizes 
    for val_batch in val_loader:
        source, target = process_batch(val_batch)
        break # only one example is needed 

    torch.onnx.export(model, 
                    (source, target), 
                    "paraphraser.onnx", 
                    opset_version=11,
                    input_names = ['input'],  
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0 : 'padded_size', 1: 'padded_size'}, 
                                'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load("paraphraser.onnx")
    onnx.checker.check_model(onnx_model)

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

    train_data_source, train_data_target = read(train_path)
    val_data_source, val_data_target = read(val_path)

    train_dataset = SentenceDataset(train_data_source, train_data_target, vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, shuffle=True)

    val_dataset = SentenceDataset(val_data_source, val_data_target, vocab)
    val_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    print('Started training... ')
    train(model, vocab, train_loader, val_loader, optimizer, embed_size, hidden_size, epochs, save_model, save_every, display_every, device)
    convert_onnx(model.load(save_model, device), val_loader, vocab) 
if __name__ == '__main__':
    app.run(main)


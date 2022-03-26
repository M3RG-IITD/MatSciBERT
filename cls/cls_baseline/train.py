import ipdb; import os; import sys
sys.path.append("my_models/")
import numpy as np; import spacy
import torch; from torchtext import data,datasets; import torch.nn as nn; import torch.optim as optim
from models import RNN; import custom_lstm; from my_dataloader import *; from my_utils import *
import time; import argparse; import copy; import math; import params; from tqdm import tqdm


global logf
def myprint(s):
    global logf
    if args.log :
        print(s)
    logf.write(str(s) +'\n')
    logf.flush()
    return


parser = params.parse_args()
args = parser.parse_args()
args = add_config(args) if args.config_file != None else args
assert(args.mode == 'train' or args.mode == 'resume')

set_all_seeds_to(args.seed)

MAX_VOCAB_SIZE = 25000 if (args.cap_vocab) else 100000
print(MAX_VOCAB_SIZE)

device = torch.device('cuda:{0}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
if args.pool == 'last1' or args.pool == 'max1' or args.pool == 'mean1':
    custom_lstm.forget_bias = args.forget_bias

args.model_path = get_model_path(args)
model_dir = '../models/' + args.task + '/' + args.pool + '/' + args.model_path if args.seed == 1234 else f'../models_{str(args.seed)}/' + args.task + '/' + args.pool + '/' + args.model_path
print(model_dir)

model_name = model_dir + '/best.pt'
if args.mode == 'resume':
    print('Resume')
    model_name = model_dir + '/best_resume.pt'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logf, train_acc_f, valid_acc_f, test_acc_f, ratios_f = get_all_logs(args,model_dir)

TEXT, LABEL, train_iterator, valid_iterator, test_iterator = get_data(args, MAX_VOCAB_SIZE, device)
myprint('Data Loading done!')

vocab_size = len(TEXT.vocab) 
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
output_dim = len(LABEL.vocab)

model = RNN(
    vocab_size = vocab_size,
    embedding_dim = args.embed_dim, 
    hidden_dim = args.hidden_dim, 
    output_dim = output_dim, 
    bidirectional = args.bidirectional, 
    pad_idx = pad_idx, 
    gpu_id = args.gpu_id, 
    pool = args.pool, 
    percent = None, 
    pos_vec = "none", 
    pos_wiki= "none", 
    dc = args.drop_connect, 
    customlstm=args.customlstm, 
    num_layers = args.num_layers,
)


if args.glove and args.use_embedding:
    pretrained_embeddings = TEXT.vocab.vectors
    myprint(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)

if args.freeze_embedding:
    model.embedding.weight.requires_grad = False
    
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

myprint(f'The model has {count_parameters(model):,} trainable parameters')

if args.optimizer == 'SGD':
    print('Using SGD')
    optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay, 
                            lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
else:    
    print('Using Adam')
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, amsgrad=args.amsgrad)

criterion = nn.CrossEntropyLoss().to(device)
accuracy = categorical_accuracy


def iter_func(iterator):
    if args.log:
        return tqdm(iterator)
    else:
        return iterator


def train(model, iterator, optimizer, criterion, epoch, valid_iterator):
    global sum_norm, num_points, all_gradients, all_activations
    epoch_loss = 0
    model.train()
    n = 0
    sum_norm = 0
    num_points = 20
    all_gradients, all_activations = [], []
    copy_train_iterator = copy.copy(train_iterator)
    total_correct = 0
    
    for batch in iter_func(iterator):
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths, gradients=args.gradients, use_embedding=args.use_embedding)[0].squeeze(1)
        loss = criterion(predictions, batch.label)
        correct, tot = accuracy(predictions, batch.label)
        total_correct += correct
        n += tot
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), total_correct / n


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    total_correct = 0
    model.eval()
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            text, text_lengths = batch.text
            predictions = model(text, text_lengths, use_embedding = args.use_embedding)[0].squeeze(1)
            loss = criterion(predictions, batch.label)
            correct, tot = accuracy(predictions, batch.label)
            total_correct += correct
            epoch_loss += loss.item()
            n += tot
    return epoch_loss / len(iterator), total_correct / n


funct = train
funce = evaluate 
epoch_initial = 0

if args.mode == 'resume':
    checkpoint = torch.load(model_dir + '/final.pt', map_location = device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    epoch_initial = checkpoint['epoch']

best_valid_acc = 0
final_valid_loss = 0
patience_max = 20
patience = 0
if args.gradients == True:
	patience_max = 100

for epoch in range(epoch_initial, args.epochs+epoch_initial):
    start_time = time.time()
    train_loss, train_acc = funct(model, train_iterator, optimizer, criterion, epoch , valid_iterator)
    valid_loss, valid_acc = funce(model, valid_iterator, criterion)
    if valid_acc < best_valid_acc:
        patience +=1
    else:
        patience = 0

    final_valid_loss = valid_loss 
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    }, model_name)

    myprint(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    myprint(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    myprint(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    train_acc_f.write(str(train_acc*100)+'\n')
    valid_acc_f.write(str(valid_acc*100)+'\n')
    train_acc_f.flush()
    valid_acc_f.flush()

    if patience == patience_max:
        break

torch.save({
    'epoch' : args.epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': final_valid_loss,
    }, model_dir + '/final.pt')
    
ratios_f.close() if args.gradients else None
train_acc_f.close()


checkpoint = torch.load(model_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

val_loss, val_acc = evaluate(model, valid_iterator, criterion)
myprint(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}%')
valid_acc_f.write(str(val_acc * 100) + '\n')
valid_acc_f.flush()
valid_acc_f.close()

test_loss, test_acc = evaluate(model, test_iterator, criterion)
myprint(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
test_acc_f.write(str(test_acc * 100) + '\n')
test_acc_f.flush()
test_acc_f.close()

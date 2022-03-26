import torch
from torchtext import data, datasets
import torch.nn as nn
import torch.optim as optim
import dill as pickle
import numpy as np
import random


def load_pickle(args, TEXT, LABEL):
    train_list, valid_list, test_list = pickle.load(open('../datasets/'+args.task+f'/dump.pkl', 'rb'))
    fields={'Abstract': ('text', TEXT), 'Label': ('label', LABEL)}
    fields, field_dict = [], fields
    for field in field_dict.values():
        if isinstance(field, list):
            fields.extend(field)
        else:
            fields.append(field) 
    train_data, valid_data, test_data = data.Dataset(train_list, fields=fields), data.Dataset(valid_list, fields=fields), data.Dataset(test_list, fields=fields)
    return train_data, valid_data, test_data


def dump_pickle(args, TEXT, LABEL):
    train_path = 'train.csv'
    test_path = 'test.csv'
    dev_path = 'val.csv'
   
    task = args.task # glass_non_glass
    print('WARNING: Pickle Load Unsuccessful. Training time will increase')

    train_data, valid_data, test_data = data.TabularDataset.splits(
            path='../datasets/'+task, train=train_path,
            validation=dev_path, test=test_path, format='csv',
            fields={'Abstract': ('text', TEXT), 'Label': ('label', LABEL)})

    train_list, valid_list, test_list = list(train_data), list(valid_data), list(test_data)
    random.shuffle(train_list); random.shuffle(valid_list); random.shuffle(test_list)
    pickle.dump([train_list, valid_list, test_list], open('../datasets/'+task+f'/dump.pkl', 'wb'))


def get_data(args, MAX_VOCAB_SIZE, device):
    task = args.task
    print(task)
    tokenizer = data.utils.get_tokenizer('spacy', language='en_core_web_sm')
    TEXT = data.Field(tokenize=tokenizer, include_lengths=True) 
    LABEL = data.LabelField()

    try:
        train_data, valid_data, test_data = load_pickle(args, TEXT, LABEL)
    except:
        dump_pickle(args, TEXT, LABEL)
        train_data, valid_data, test_data = load_pickle(args, TEXT, LABEL)

    LABEL.build_vocab(train_data)
    
    vec = 'glove.6B.100d' if args.glove else None
    unk_init = torch.Tensor.normal_ if args.glove else None

    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors=vec, unk_init=unk_init)    
    train_iterator, valid_iterator, test_iterator \
                                = data.BucketIterator.splits((train_data, valid_data, test_data),     
                                                batch_size=args.batch_size, 
                                                sort_within_batch=True, 
                                                sort_key=lambda x: len(x.text),
                                                device=device)

    return TEXT, LABEL, train_iterator, valid_iterator, test_iterator


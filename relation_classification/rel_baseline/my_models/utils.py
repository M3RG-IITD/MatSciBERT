import os
# import ipdb
import argparse
import json
import sys 
import torch
from string import ascii_lowercase

import torch.nn as nn
import torch.nn.functional as F
from custom_lstm import *

def swish(x):
    return x * F.sigmoid(x)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class gelu_module(nn.Module):
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class swish_module(nn.Module):
    def forward(self,x):
        return x * F.sigmoid(x)

def total_extra_params(heads, config):
    if (config.use_base):
        head_inp_dim = 2*config.hidden_dim
    
    if config.encoder_type == 'MultiMax':
        seq_in_size = 4 #for prem, hypo ..
        if(config.use_base):
            seq_in_size *= (heads*config.final_dim) + config.hidden_dim*2
        else:
            seq_in_size *= (heads*config.final_dim)
        #############MODIFIED#########################

        if(config.maxout == 'min_max'):
            seq_in_size *= 2 
        if(config.maxout == 'avg_max'):
            seq_in_size *= 2 
        if(config.interactive == True):
            seq_in_size *= 2             

    total_params_mlp = (seq_in_size+1)*config.fc_dim + (config.fc_dim+1)*config.fc_dim + (config.fc_dim+1)*config.out_dim
    total_params_Mw = heads*((head_inp_dim+1)*config.hidden_dim + (config.hidden_dim+1)*config.final_dim)
    total_params_extra = total_params_Mw + total_params_mlp
    return total_params_extra

class MyRNN(nn.Module):

    def __init__(self, config):
        super(MyRNN, self).__init__()
        # total_params = 6168*(config.hidden_dim) 

        heads_0 = 0
        total_params_with_h_heads = total_extra_params(config.heads, config)
        total_params_with_10_heads = total_extra_params(10, config)

        overhead = total_params_with_10_heads - total_params_with_h_heads

        # total_params = 6200*hidden
        # hidden = int((100*6200 + overhead)/6200)
        hidden = config.hidden_dim


        rnn_params = dict(input_size=config.embed_dim, hidden_size=hidden, num_layers=config.layers, dropout=config.dropout, bidirectional=True)
        if(config.rnn == 'lstm'):
            self.rnn = nn.LSTM(**rnn_params)  
            total_params = sum(p.numel() for p in self.rnn.parameters() if p.requires_grad) + total_params_with_h_heads
            print(total_params)
            # ipdb.set_trace()      
        if(config.rnn == 'custom_lstm'):
            emb_size = config.embed_dim
            h = config.hidden_dim
            self.rnn = CustomLSTM(emb_size, h, config) 
            # params.bidirectional
            # params.cuda
        elif(config.rnn == 'gru'):
            self.rnn = nn.GRU(**rnn_params)
        elif(config.rnn == 'rnn_tanh'):
            self.rnn = nn.RNN(**rnn_params)
        elif(config.rnn == 'rnn_relu'):
            self.rnn = nn.RNN(**rnn_params, nonlinearity='relu')

    def forward(self, inputs):
        out, hn = self.rnn(inputs)
        return out

def tensorize(embs):
    list_embs = []
    for i in range(len(embs['features'])):
        list_embs.append(embs['features'][i]['layers'][0]['values'])
    tokens = [f['token'] for f in embs['features']]
    sep_index = tokens.index('[SEP]')
    list_embs1 = list_embs[:sep_index+1]
    tokens1 = tokens[:sep_index+1]
    list_embs2 = list_embs[sep_index+1:]
    tokens2 = tokens[sep_index+1:]

    tensor_embs1 = torch.FloatTensor(list_embs1)
    tensor_embs2 = torch.FloatTensor(list_embs2)
    return tensor_embs1, tensor_embs2, tokens1, tokens2

def to_cuda(v, device):
    if(torch.cuda.is_available()):
        return v.to(device)
    else:
        return v

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str)
    parser.add_argument('--action', type=str)
    parser.add_argument('--model', type=str)

    return parser


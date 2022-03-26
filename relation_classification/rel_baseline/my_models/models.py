import ipdb
import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
from custom_lstm import CustomLSTM
import torch.nn.functional as F
import numpy as np
import sys


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 bidirectional, pad_idx, gpu_id, pool, percent, pos_vec = "none", pos_wiki="none", dc=0, customlstm = 0,
                 num_layers = 1):
        super().__init__()
        self.num_layers = num_layers
        if embedding_dim > 1:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.device = 'cuda:{0}'.format(gpu_id) if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.custom_lstm = customlstm
        if self.custom_lstm:
            print("Using Custom LSTM")
            self.rnn = CustomLSTM( embedding_dim, hidden_dim, bidirectional=bidirectional, cuda = self.device,dc=dc)
        else:
            print("Using Standard LSTM")
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first = True,
                                num_layers = num_layers)        
        self.pos_wiki = pos_wiki
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)
        self.pool = pool
        self.pos_vec = pos_vec
        self.percent = percent
        if pool in ["att_luong"] :
            self.att_vec = nn.Linear(hidden_dim*num_directions, 1, bias = False)

        self.pooling_func = {   
                                "max":          self.max_pool,
                                "mean":         self.mean_pool,
                                "min":          self.min_pool,
                                "att_luong":    self.attention_luong,
                                "att_max":      self.attention_max,  
                            }

        if pool[-1] == "1" and customlstm == 0:
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n,  names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1.)

    def mean_pool(self, hidden, cell):
        return hidden.mean(dim = 1), None

    def max_pool(self, hidden, cell):
        batch_size, seq_len = hidden.shape[0],hidden.shape[1]
        max_vals = hidden.max(dim = 1)
        # ipdb.set_trace()
        indices = max_vals[1]
        #full_tensor is made so that if there is a position that never occurred -- that is also accounted for.
        full_tensor = torch.arange(seq_len).to(self.device).unsqueeze(0).repeat(batch_size,1) 
        final_indices = torch.cat((indices,full_tensor), dim = 1) 
        x_unique = final_indices.unique(sorted=True, return_counts=True) 
        counts = (x_unique[1] - batch_size)
        weights = counts.float()/counts.sum()
        return max_vals[0], weights.unsqueeze(0)
    
    def min_pool(self, hidden, cell):
        return hidden.min(dim = 1)[0], None

    def attention_luong(self, hidden, cell):
        # att_vec = [hidden_dim,1] # hidden = [batch_size, seq_len, hidden_dim]
        dot_prod    = self.att_vec(hidden).squeeze(2)                   # dot_prod = [batch_size, seq_len]
        weights     = torch.softmax(dot_prod, dim = 1).unsqueeze(1)     # weights = [batch_size, 1, seq_len]
        sent_emb    = torch.matmul(weights,hidden).squeeze(1)           # sent_emb = [batch_size, hidden_dim]
        return sent_emb, weights

    def attention_max(self, hidden, cell):
        c2 = hidden.permute(0,2,1)                              # c2 = [batch_size, 2*hidden_state, sent_len]
        c2_n = c2/c2.norm(dim=1).unsqueeze(1)                   # normalize to get direction                                    
        q1  = hidden.max(dim = 1)[0].unsqueeze(1)               # query = [batch_size, 1, hidden_dim]
        bmm = torch.bmm(q1, c2_n)                               # bmm = [batch_size, 1, sent_len]
        weights = torch.softmax(bmm, dim = 2)                   # weights = [batch_size, 1, sent_len]
        sent_emb = torch.matmul(weights,hidden).squeeze(1)      # sent_emb = [batch_size, 2*hidden_state]
        return sent_emb, weights


    def forward(self, text, text_lengths, gradients=False, explain = False, return_attention_weights = False, dropout = 0, use_embedding = True):

        word_emb = self.embedding(text) if use_embedding else text  # word_emb = [sent len, batch size, emb dim]                      
        sent_len = word_emb.shape[0]
        batch_size = word_emb.shape[1]
        dims = word_emb.shape[2]
        percent = self.percent
            
        word_emb = word_emb.permute(1,0,2)
        if self.custom_lstm:
            (hidden, cell) = self.rnn(word_emb, gradients)  
        else:
            (hidden, cell) = self.rnn(word_emb)

        if self.pool in ['last', 'last1']:
            if self.num_layers == 1:
                sent_emb = cell[0] if self.custom_lstm else cell[0].permute(1,2,0).reshape(batch_size,-1)
            else:
                sent_emb = hidden[:,-1,:]
        elif self.pool == 'att_drop':
            sent_emb, weights = self.attention_drop(hidden,cell,dropout)
        else:
            sent_emb, weights = self.pooling_func[self.pool](hidden,cell)
        if return_attention_weights:
            return self.fc(sent_emb), weights.squeeze(1).mean(dim = 0)
        return self.fc(sent_emb), sent_emb, word_emb


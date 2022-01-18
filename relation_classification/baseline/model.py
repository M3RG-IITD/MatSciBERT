import torch
from torch import nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=50000, emb_dim=100, emb_vectors=None,
                 emb_dropout=0.3,
                 lstm_dim=256, lstm_n_layer=2, lstm_dropout=0.3,
                 bidirectional=True, lstm_combine='add',
                 n_linear=2, linear_dropout=0.5, n_classes=1,
                 crit=nn.CrossEntropyLoss()):
        super().__init__()
        vocab_size, emb_dim = emb_vectors.shape
        n_dirs = bidirectional + 1
        lstm_dir_dim = lstm_dim // n_dirs if lstm_combine == 'concat' else lstm_dim

        self.lstm_n_layer = lstm_n_layer
        self.n_dirs = n_dirs
        self.lstm_dir_dim = lstm_dir_dim
        self.lstm_combine = lstm_combine

        self.embedding_layer = nn.Embedding(*emb_vectors.shape)
        self.embedding_layer.from_pretrained(emb_vectors, padding_idx=1)
        # pad=1 in torchtext; embedding weights trainable
        self.embedding_dropout = nn.Dropout(p=emb_dropout)

        self.lstm = nn.LSTM(emb_dim, lstm_dir_dim,
                            num_layers=lstm_n_layer,
                            bidirectional=bidirectional,
                            batch_first=True)
        if lstm_n_layer > 1: self.lstm.dropout = lstm_dropout
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

        self.att_w = nn.Parameter(torch.randn(1, lstm_dim, 1))
        self.linear_layers = [nn.Linear(lstm_dim, lstm_dim) for _ in
                              range(n_linear - 1)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.linear_dropout = nn.Dropout(p=linear_dropout)

        self.label = nn.Linear(lstm_dim, n_classes)
        self.crit = crit

        self.opts = {
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'emb_dropout': emb_dropout,
            'emb_vectors': emb_vectors,
            'lstm_dim': lstm_dim,
            'lstm_n_layer': lstm_n_layer,
            'lstm_dropout': lstm_dropout,
            'lstm_combine': lstm_combine,
            'n_linear': n_linear,
            'linear_dropout': linear_dropout,
            'n_classes': n_classes,
            'crit': crit,
        }

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(
            2)
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(
            2)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.view(self.lstm_n_layer, self.n_dirs, batch_size,
                               self.lstm_dir_dim)[-1]
        final_h = final_h.permute(1, 0, 2)
        final_h = final_h.sum(dim=1)  # (batch_size, 1, self.half_dim)

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        if self.lstm_combine == 'add':
            lstm_output = lstm_output.view(batch_size, seq_len, 2,
                                           self.lstm_dir_dim)
            lstm_output = lstm_output.sum(dim=2)
            # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input):
        batch_size, seq_len, *_ = input.shape
        inp = self.embedding_layer(input)
        inp = self.embedding_dropout(inp)

        lstm_output, (final_h, final_c) = self.lstm(inp)

        # outputs = []
        # for i in range(seq_len):
        #     cur_emb = inp[i:i + 1]  # .view(1, inp.size(1), inp.size(2))
        #
        #     o, hidden = self.lstm(cur_emb) if i == 0 else self.lstm(cur_emb, hidden)
        #     import pdb;pdb.set_trace()
        #     outputs += [o.unsqueeze(0)]
        #
        # outputs = torch.cat(outputs, dim=0)

        lstm_output = self.lstm_dropout(lstm_output)

        attn_output = self.re_attention(lstm_output, final_h, input)
        output = self.linear_dropout(attn_output)

        for layer in self.linear_layers:
            output = layer(output)
            output = self.linear_dropout(output)
            output = F.relu(output)

        logits = self.label(output)
        return logits

    def forward_normal_attention(self):
        batch_size = len(input)

        inp = self.embedding_layer(input)
        inp = self.embedding_dropout(inp)
        lstm_output, (final_h, final_c) = self.lstm(inp)
        final_h = final_h.view(self.lstm_n_layer, self.n_dirs, batch_size,
                               self.lstm_dim // self.n_dirs)[-1]
        final_h = final_h.permute(1, 0,
                                  2)  # (batch_size, 2, self.lstm_dim // self.n_dirs)
        final_h = final_h.contiguous().view(batch_size, self.lstm_dim)

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(lstm_output, final_h)

        output = self.linear_dropout(attn_output)

        for layer in self.linear_layers:
            output = layer(output)
            output = self.linear_dropout(output)
            output = F.relu(output)

        logits = self.label(output)
        return logits

    def forward_normal_lstm(self):
        inp = self.embedding_layer(input)
        inp = self.embedding_dropout(inp)
        lstm_output, (final_h, final_c) = self.lstm(inp)
        # output.size() = (batch_size, num_seq, hidden_size)

        output = lstm_output[:, -1]
        output = self.linear_dropout(output)

        for layer in self.linear_layers:
            output = layer(output)
            output = self.linear_dropout(output)
            output = F.relu(output)

        logits = self.label(output)
        return logits

    def loss(self, input, target):
        logits = self.forward(input)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.crit(logits_flat, target_flat)  # mean_score per batch
        return loss

    def predict(self, input):
        logits = self.forward(input)
        logits[:, :2] = float('-inf')
        preds = logits.max(dim=-1)[1]
        preds = preds.detach().cpu().numpy().tolist()
        return preds

    def loss_n_acc(self, input, target):
        logits = self.forward(input)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.crit(logits_flat, target_flat)  # mean_score per batch

        pred_flat = logits_flat.max(dim=-1)[1]
        acc = (pred_flat == target_flat).sum()
        return loss, acc.item()

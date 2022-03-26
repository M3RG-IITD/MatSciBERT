from __future__ import print_function

import os
import ipdb
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import _VF

import utils
import numpy as np
import pickle

import random

def round_down(num, divisor):
    return num - (num%divisor)

forget_bias = None 
class CustomLSTM(nn.Module):
    """
     Attributes:
        weight_ih: the learnable input-hidden weights, of shape `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`
        ht or hx -> h value per time step
        ct or cx -> c (forwarded to next cell) value per time step
    """
    def __init__(self, emb_size, h, bidirectional, cuda = 'cuda:0', dc=0):
        super(CustomLSTM, self).__init__()
        self.emb_size = emb_size
        self.h = h
        self.forward_cell = nn.LSTMCell(emb_size, h)
        self.backward_cell = nn.LSTMCell(emb_size, h)

        if(forget_bias != None):
            hidden_size = h
            self.forward_cell.bias_hh.data[hidden_size:2*hidden_size].fill_(forget_bias)
            self.forward_cell.bias_ih.data[hidden_size:2*hidden_size].fill_(forget_bias)
            self.backward_cell.bias_ih.data[hidden_size:2*hidden_size].fill_(forget_bias)
            self.backward_cell.bias_hh.data[hidden_size:2*hidden_size].fill_(forget_bias)

        self.f_hh, self.f_ih, self.f_ht = [], [], []
        self.b_hh, self.b_ih, self.b_ht = [], [], []
        self.grad_fi, self.grad_ff, self.grad_fo = [], [], []
        self.grad_bi, self.grad_bf, self.grad_bo = [], [], []

        self.forward_acts= []
        self.backward_acts= []
        self.hh_norms, self.ih_norms, self.ht_norms = [], [], []
        self.cnt = 0
        self.cuda = cuda
        self.bidirectional = bidirectional
        self.dc = dc

    def concatenate_lists(self, l1, l2):
        length = len(l1)
        ret_list = []
        for i in range(length):
            ret_list.append(torch.norm(torch.cat((l1[i],l2[i]), dim = 1)))
        return ret_list

    def concatenate_lists_internal(self, l1, l2):
        length = len(l1)
        ret_list = []
        for i in range(length):
            ret_list.append(torch.cat((l1[i],l2[i]), dim = 2))
        return ret_list

    def get_norms(self, l1):
        length = len(l1)
        ret_list = []
        for i in range(length):
            ret_list.append(torch.norm(l1[i]))
        return ret_list

    def save_fhh(self, in_grad):
        self.f_hh.append((in_grad))
        self.grad_fi.append(in_grad[:self.h].norm().item())
        self.grad_ff.append(in_grad[self.h:2*self.h].norm().item())
        # print(in_grad[2*self.h:3*self.h].norm().item())
        self.grad_fo.append(in_grad[3*self.h:].norm().item())
        return

    def save_bhh(self, in_grad):
        self.b_hh= [(in_grad)] + self.b_hh
        self.grad_bi.append(in_grad[:self.h].norm().item())
        self.grad_bf.append(in_grad[self.h:2*self.h].norm().item())
        self.grad_bo.append(in_grad[3*self.h:].norm().item())
        # self.b_hh.append((in_grad))
        return

    def save_fih(self, in_grad):
        self.f_ih.append((in_grad))
        return

    def save_bih(self, in_grad):
        self.b_ih= [(in_grad)] + self.b_ih
        # self.b_ih.append((in_grad))
        return

    def save_fht(self, in_grad):
        self.f_ht.append((in_grad))
        return

    def save_bht(self, in_grad):
        self.b_ht= [(in_grad)] + self.b_ht
        return

    def drop_connect(self, x):
        dropout = self.dc
        if not self.training or dropout == 0:
            return x
        m = x.data.new(x.size(0), x.size(1)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        # mask = mask.expand_as(x)
        return mask * x

    def get_activations(self, inputs, hx, weight_hh, weight_ih, bias_hh, bias_ih):
        Wih, Wfh, _, Woh = weight_hh[:self.h], weight_hh[self.h:2*self.h], weight_hh[2*self.h:3*self.h], weight_hh[-self.h:]
        Wii, Wfi, _, Woi = weight_ih[:self.h], weight_ih[self.h:2*self.h], weight_ih[2*self.h:3*self.h], weight_ih[-self.h:]
        bih, bfh, _, boh = bias_hh[:self.h], bias_hh[self.h:2*self.h], bias_hh[2*self.h:3*self.h], bias_hh[-self.h:]
        bii, bfi, _, boi = bias_ih[:self.h], bias_ih[self.h:2*self.h], bias_ih[2*self.h:3*self.h], bias_ih[-self.h:]        

        i_act = torch.sigmoid(torch.matmul(hx,Wih) + torch.matmul(inputs,Wii.transpose(0,1)) + bih + bii).norm()
        f_act = torch.sigmoid(torch.matmul(hx,Wfh) + torch.matmul(inputs,Wfi.transpose(0,1)) + bfh + bfi).norm()
        o_act = torch.sigmoid(torch.matmul(hx,Woh) + torch.matmul(inputs,Woi.transpose(0,1)) + boh + boi).norm()

        return (i_act, f_act, o_act)
         

    def forward(self, inputs, gradients=False):
        batch_size = inputs.shape[0]
        f_hx =  utils.to_cuda(Variable(torch.zeros(batch_size, self.h), requires_grad=True), self.cuda)
        b_hx =  utils.to_cuda(Variable(torch.zeros(batch_size, self.h), requires_grad=True), self.cuda)
        f_cx =  utils.to_cuda(Variable(torch.zeros(batch_size, self.h), requires_grad=True), self.cuda)
        b_cx =  utils.to_cuda(Variable(torch.zeros(batch_size, self.h), requires_grad=True), self.cuda)
        f_h_list = []
        b_h_list = []
        # biash_hh = self.drop_connect(bias_hh)

        if not gradients:
            weight_ih = self.forward_cell.weight_ih.view(self.forward_cell.weight_ih.size())
            weight_hh = self.forward_cell.weight_hh.view(self.forward_cell.weight_hh.size())
            bias_ih = self.forward_cell.bias_ih.view(self.forward_cell.bias_ih.size())
            bias_hh = self.forward_cell.bias_hh.view(self.forward_cell.bias_hh.size())
            weight_hh = self.drop_connect(weight_hh)

        for i in range(inputs.shape[1]):
            if gradients:
                weight_ih = self.forward_cell.weight_ih.view(self.forward_cell.weight_ih.size())
                weight_hh = self.forward_cell.weight_hh.view(self.forward_cell.weight_hh.size())
                bias_ih = self.forward_cell.bias_ih.view(self.forward_cell.bias_ih.size())
                bias_hh = self.forward_cell.bias_hh.view(self.forward_cell.bias_hh.size())
                weight_hh = self.drop_connect(weight_hh)
            
                self.forward_acts.append(self.get_activations(inputs[:,i,:],f_hx, weight_hh, weight_ih, bias_hh, bias_ih))

            f_hx, f_cx = _VF.lstm_cell(inputs[:, i, :], (f_hx, f_cx), weight_ih, weight_hh,\
                    bias_ih, bias_hh)
            if(gradients == True):
                weight_hh.register_hook(self.save_fhh)
                weight_ih.register_hook(self.save_fih)
                try:
                    f_hx.register_hook(self.save_fht)
                except:
                    var = 0
            # hx, cx = self.cell(inputs[:, i, :], (hx, cx))
            # hx, cx = self.LSTMCell(inputs[:, i, :], (hx, cx), self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, hook=hook)
            f_h_list.append(f_hx.unsqueeze(1))
        self.hh = self.f_hh
        self.ht = self.f_ht 
        self.ih = self.f_ih
        h_list = f_h_list
        (hx,cx) = (f_hx,f_cx)

        if (self.bidirectional):
            if not gradients:
                weight_ih = self.backward_cell.weight_ih.view(self.backward_cell.weight_ih.size())
                weight_hh = self.backward_cell.weight_hh.view(self.backward_cell.weight_hh.size())
                bias_ih = self.backward_cell.bias_ih.view(self.backward_cell.bias_ih.size())
                bias_hh = self.backward_cell.bias_hh.view(self.backward_cell.bias_hh.size())
                weight_hh = self.drop_connect(weight_hh)
                # biash_hh = self.drop_connect(bias_hh)
            for i in range(inputs.shape[1] - 1, -1, -1):
                if gradients:
                    weight_ih = self.backward_cell.weight_ih.view(self.backward_cell.weight_ih.size())
                    weight_hh = self.backward_cell.weight_hh.view(self.backward_cell.weight_hh.size())
                    bias_ih = self.backward_cell.bias_ih.view(self.backward_cell.bias_ih.size())
                    bias_hh = self.backward_cell.bias_hh.view(self.backward_cell.bias_hh.size())
                    weight_hh = self.drop_connect(weight_hh)
                
                    self.backward_acts.append(self.get_activations(inputs[:,i,:],b_hx, weight_hh, weight_ih, bias_hh, bias_ih))
                    
                b_hx, b_cx = _VF.lstm_cell(inputs[:, i, :], (b_hx, b_cx), weight_ih, weight_hh,\
                        bias_ih, bias_hh)

                if(gradients == True):
                    weight_hh.register_hook(self.save_bhh)
                    weight_ih.register_hook(self.save_bih)
                    try:
                        b_hx.register_hook(self.save_bht)
                    except:
                        var = 0
                b_h_list.append(b_hx.unsqueeze(1))

            h_list = self.concatenate_lists_internal(f_h_list, b_h_list[::-1])
            (hx,cx) = (torch.cat((f_hx,b_hx), dim = 1), torch.cat((f_cx,b_cx), dim = 1))
        return torch.cat(h_list, dim=1), (hx,cx)


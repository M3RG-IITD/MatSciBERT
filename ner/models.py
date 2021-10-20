import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForTokenClassification

from torchcrf import CRF


model_revision = 'main'


class BIO_Tag_CRF(CRF):
    def __init__(self, num_tags: int, device, batch_first: bool = False):
        super(BIO_Tag_CRF, self).__init__(num_tags=num_tags, batch_first=batch_first)
        self.device = device
        start_transitions = self.start_transitions.clone().detach()
        transitions = self.transitions.clone().detach()
        assert num_tags % 2 == 1
        num_uniq_labels = (num_tags - 1) // 2
        for i in range(num_uniq_labels, 2 * num_uniq_labels):
            start_transitions[i] = -10000
            for j in range(0, num_tags):
                if j == i or j + num_uniq_labels == i: continue
                transitions[j, i] = -10000
        self.start_transitions = nn.Parameter(start_transitions)
        self.transitions = nn.Parameter(transitions)

    def forward(self, logits, labels, masks):
        
        new_logits, new_labels, new_attention_mask = [], [], []
        for logit, label, mask in zip(logits, labels, masks):
            new_logits.append(logit[mask])
            new_labels.append(label[mask])
            new_attention_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.uint8, device=self.device))
        
        padded_logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)
        padded_attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        loss = -super(BIO_Tag_CRF, self).forward(padded_logits, padded_labels, mask=padded_attention_mask, reduction='mean')
        
        if self.training:
            return (loss, )
        else:
            out = self.decode(padded_logits, mask=padded_attention_mask)
            assert(len(out) == len(labels))
            out_logits = torch.zeros_like(logits)
            for i in range(len(out)):
                k = 0
                for j in range(len(labels[i])):
                    if labels[i][j] == -100: continue
                    out_logits[i][j][out[i][k]] = 1.0
                    k += 1
                assert(k == len(out[i]))
            return (loss, out_logits, )


class BERT_CRF(nn.Module):
    def __init__(self, model_name, device, config, cache_dir):
        super(BERT_CRF, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        assert('labels' in inputs)
        logits = self.encoder(**inputs)[1]
        labels = inputs['labels']
        masks = (labels != -100)
        return self.crf(logits, labels, masks)


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, model_name, device, config, cache_dir, hidden_dim):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(config.hidden_size, hidden_dim, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * hidden_dim, config.num_labels)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)
    
    def forward(self, **inputs):
        assert('labels' in inputs)
        last_hidden_state = self.encoder(output_hidden_states=True, **inputs)[2][-1]
        out = self.lstm(self.dropout1(last_hidden_state))[0]
        logits = self.fc(self.dropout2(out))
        labels = inputs['labels']
        masks = (labels != -100)
        return self.crf(logits, labels, masks)


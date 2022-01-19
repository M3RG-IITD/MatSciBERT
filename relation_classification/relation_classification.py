import copy
import os
from pathlib import Path
import pickle
import sys
sys.path.append('..')

import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import f1_score, classification_report

from normalize_text import normalize

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


parser = ArgumentParser()
parser.add_argument('--model_name', required=True, choices=['scibert', 'matscibert', 'bert'], type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--preds_save_dir', default=None, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--seeds', nargs='+', default=None, type=int)
parser.add_argument('--lm_lrs', nargs='+', default=None, type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
args = parser.parse_args()

if args.model_name == 'scibert':
    model_name = 'allenai/scibert_scivocab_uncased'
    to_normalize = False
elif args.model_name == 'matscibert':
    model_name = 'm3rg-iitd/matscibert'
    to_normalize = True
elif args.model_name == 'bert':
    model_name = 'bert-base-uncased'
    to_normalize = False
else:
    raise NotImplementedError

model_revision = 'main'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
preds_save_dir = ensure_dir(args.preds_save_dir) if args.preds_save_dir else None

if args.seeds is None:
    args.seeds = [0, 1, 2]
if args.lm_lrs is None:
    args.lm_lrs = [2e-5, 3e-5, 5e-5]

data_dir = 'datasets/annotated-materials-syntheses'


def get_data_of_split(split_name):
    f = open(os.path.join(data_dir, 'sfex-' + split_name + '-fnames.txt'))
    files = f.read().strip().split()
    f.close()
    
    X, y = [], []
    
    for file in files:
        f = open(os.path.join(data_dir, 'data', file + '.txt'))
        text = f.read()
        f.close()
        f = open(os.path.join(data_dir, 'data', file + '.ann'))
        ann = f.read().strip().split('\n')
        f.close()

        t_dict, e_dict, relations = dict(), dict(), []

        for l in ann:
            s = l.strip()
            if s[0] == 'T':
                s = s.split('\t')
                assert len(s) == 3
                idxs = s[1].split()[1:]
                idxs = (int(idxs[0]), int(idxs[1]))
                t_dict[s[0]] = idxs
                assert s[2] == text[idxs[0]:idxs[1]]
            elif s[0] == 'E':
                s = s.split('\t')
                assert len(s) == 2
                args = s[1].split(' ')
                e_dict[s[0]] = args[0].split(':')[1]
                for a in range(1, len(args)):
                    relations.append((args[a].split(':')[0], args[0].split(':')[1], args[a].split(':')[1]))
            elif s[0] == 'R':
                s = s.split('\t')
                assert len(s) == 2
                args = s[1].split(' ')
                assert len(args) == 3
                relations.append((args[0], args[1].split(':')[1], args[2].split(':')[1]))

        e_dict = {k: t_dict[v] for k, v in e_dict.items()}
        rels = []
        for r in relations:
            e1 = t_dict[r[1]] if r[1][0] == 'T' else e_dict[r[1]]
            e2 = t_dict[r[2]] if r[2][0] == 'T' else e_dict[r[2]]
            rels.append((r[0], e1, e2))

        for v in rels:
            min_idx = text.rfind('\n', 0, min(v[1][0], v[2][0])) + 1
            max_idx = text.find('\n', max(v[1][1], v[2][1]))
            if max_idx == -1: max_idx = len(text)
            if '\n' in text[min_idx:max_idx]: continue
            e1 = (v[1][0] - min_idx, v[1][1] - min_idx)
            e2 = (v[2][0] - min_idx, v[2][1] - min_idx)
            X.append((text[min_idx:max_idx], e1, e2, file))
            y.append(v[0])
    return X, y


train_X, train_y = get_data_of_split('train')
val_X, val_y = get_data_of_split('dev')
test_X, test_y = get_data_of_split('test')

print(len(train_X), len(val_X), len(test_X))

unique_labels = set(train_y)
label_list = sorted(list(unique_labels))
print(label_list)
tag2id = {tag: id for id, tag in enumerate(label_list)}
num_labels = len(label_list)

max_seq_len = 512
tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
    'model_max_length': max_seq_len
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])


def tokenize(text: str):
    if to_normalize:
        return tokenizer.tokenize(normalize(text))
    return tokenizer.tokenize(text)


def rc_tokenize(X, y):
    encodings = {'input_ids': [], 'attention_mask': [], 'entity_markers': []}
    for i in range(len(X)):
        text, (s1, e1), (s2, e2), f = X[i]
        if s1 < s2:
            tokens = tokenize(text[:s1]) + ['[E1]'] + tokenize(text[s1:e1]) + ['[/E1]'] + \
                    tokenize(text[e1:s2]) + ['[E2]'] + tokenize(text[s2:e2]) + ['[/E2]'] + \
                    tokenize(text[e2:])
        else:
            tokens = tokenize(text[:s2]) + ['[E2]'] + tokenize(text[s2:e2]) + ['[/E2]'] + \
                    tokenize(text[e2:s1]) + ['[E1]'] + tokenize(text[s1:e1]) + ['[/E1]'] + \
                    tokenize(text[e1:])

        s1 = tokens.index('[E1]')
        e1 = tokens.index('[/E1]')
        s2 = tokens.index('[E2]')
        e2 = tokens.index('[/E2]')

        if len(tokens) <= max_seq_len - 2:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        elif s1 < s2:
            rem = (max_seq_len - 2) - (e2 - s1 + 1)
            assert rem >= 0
            s = max(0, s1 - rem // 2)
            e = min(len(tokens)-1, e2 + rem // 2)
            tokens = ['[CLS]'] + tokens[s:e+1] + ['[SEP]']
        else:
            rem = (max_seq_len - 2) - (e1 - s2 + 1)
            assert rem >= 0
            s = max(0, s2 - rem // 2)
            e = min(len(tokens)-1, e1 + rem // 2)
            tokens = ['[CLS]'] + tokens[s:e+1] + ['[SEP]']
            
        encodings['entity_markers'].append([tokens.index('[E1]'), tokens.index('[E2]')])

        tokens = tokenizer.convert_tokens_to_ids(tokens)
        assert len(tokens) <= max_seq_len
        
        encodings['input_ids'].append(tokens)
        encodings['attention_mask'].append([1] * len(tokens))
        
        y[i] = tag2id[y[i]]
    return encodings, y


train_encodings, train_labels = rc_tokenize(train_X, train_y)
val_encodings, val_labels = rc_tokenize(val_X, val_y)
test_encodings, test_labels = rc_tokenize(test_X, test_y)


class RC_Dataset(torch.utils.data.Dataset):
    def __init__(self, inp, labels):
        self.inp = inp
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inp.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = RC_Dataset(train_encodings, train_labels)
val_dataset = RC_Dataset(val_encodings, val_labels)
test_dataset = RC_Dataset(test_encodings, test_labels)

config_kwargs = {
    'num_labels': num_labels,
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)


def compute_metrics(p):
    predictions, y_true = p
    y_pred = np.argmax(predictions, axis=1)
    report = classification_report(y_true, y_pred, labels=np.arange(num_labels), target_names=label_list, output_dict=True)
    results = {}
    for k in report:
        if k in label_list:
            results[k] = report[k]['f1-score']
    results['macro_f1'] = report['macro avg']['f1-score']
    results['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    return results


loss_fn = nn.CrossEntropyLoss()


class BERT_RC(nn.Module):
    def __init__(self, model_name):
        super(BERT_RC, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name, from_tf=False, config=copy.copy(config), 
                                                cache_dir=cache_dir, revision=model_revision,
                                                use_auth_token=None)
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(2*config.hidden_size, num_labels)

    def forward(self, **inputs):
        assert('labels' in inputs)
        hidden_states = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        outs = torch.cat([hidden_states[torch.arange(len(hidden_states)), inputs['entity_markers'][:, 0]],
                            hidden_states[torch.arange(len(hidden_states)), inputs['entity_markers'][:, 1]]], dim=1)
        logits = self.linear(self.dropout(outs))
        return (loss_fn(logits, inputs['labels']), logits, )


metric_for_best_model = 'macro_f1'
other_metric = 'micro_f1'

best_lr = None
best_val = 0
best_val_acc_list = None
best_test_acc_list = None
best_val_oth_list = None
best_test_oth_list = None

num_epochs = 10

for lr in args.lm_lrs:

    print(f'lr: {lr}')
    val_acc, val_oth = [], []
    test_acc, test_oth = [], []
    
    for SEED in args.seeds:
        
        print(f'SEED: {SEED}')

        torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        set_seed(SEED)

        training_args = TrainingArguments(
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            evaluation_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            save_total_limit=2,
            warmup_ratio=0.1,
            learning_rate=lr,
            seed=SEED
        )
        
        model = BERT_RC(model_name).to(device)
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': args.non_lm_lr},
            {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': lr}
        ]
        optimizer_kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        }
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
        )

        train_result = trainer.train()
        print(train_result)

        val_result = trainer.evaluate()
        print(val_result)
        val_acc.append(val_result[f'eval_{metric_for_best_model}'])
        val_oth.append(val_result[f'eval_{other_metric}'])

        test_result = trainer.evaluate(test_dataset)
        print(test_result)
        test_acc.append(test_result[f'eval_{metric_for_best_model}'])
        test_oth.append(test_result[f'eval_{other_metric}'])

        if preds_save_dir:
            val_preds = trainer.predict(val_dataset).predictions
            test_preds = trainer.predict(test_dataset).predictions

            for split, preds in zip(['val', 'test'], [val_preds, test_preds]):
                file_path = os.path.join(preds_save_dir, f'annotated-materials-syntheses/{split}_{args.model_name}_{lr}_{SEED}.pkl')
                ensure_dir(os.path.dirname(file_path))
                pickle.dump(preds, open(file_path, 'wb'))

    if np.mean(val_acc) > best_val:
        best_val = np.mean(val_acc)
        best_lr = lr
        best_val_acc_list = val_acc
        best_test_acc_list = test_acc
        best_val_oth_list = val_oth
        best_test_oth_list = test_oth


print(args.model_name)
print(f'best_lr: {best_lr}')
print(f'best_val: {best_val}')
print(f'best_val {metric_for_best_model}: {best_val_acc_list}')
print(f'best_test {metric_for_best_model}: {best_test_acc_list}')
print(f'best_val {other_metric}: {best_val_oth_list}')
print(f'best_test {other_metric}: {best_test_oth_list}')

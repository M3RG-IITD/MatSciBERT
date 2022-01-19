import os
from pathlib import Path
import pickle

from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
from torch import nn

import ner_datasets
from models import BERT_CRF, BERT_BiLSTM_CRF
import conlleval

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
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
parser.add_argument('--model_name', choices=['scibert', 'matscibert', 'bert'], required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--preds_save_dir', default=None, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--seeds', nargs='+', default=None, type=int)
parser.add_argument('--lm_lrs', nargs='+', default=None, type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
parser.add_argument('--architecture', choices=['bert', 'bert-crf', 'bert-bilstm-crf'], required=True, type=str)
parser.add_argument('--dataset_name', choices=['sofc', 'sofc_slot', 'matscholar'], required=True, type=str)
parser.add_argument('--fold_num', default=None, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
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

dataset_name = args.dataset_name
fold_num = args.fold_num
model_revision = 'main'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
preds_save_dir = ensure_dir(args.preds_save_dir) if args.preds_save_dir else None
if preds_save_dir:
    preds_save_dir = os.path.join(preds_save_dir, dataset_name)
    if fold_num:
        preds_save_dir = os.path.join(preds_save_dir, f'cv_{fold_num}')
    preds_save_dir = ensure_dir(preds_save_dir)

if args.seeds is None:
    args.seeds = [0, 1, 2]
if args.lm_lrs is None:
    args.lm_lrs = [2e-5, 3e-5, 5e-5]

train_X, train_y, val_X, val_y, test_X, test_y = ner_datasets.get_ner_data(dataset_name, fold=fold_num, norm=to_normalize)
print(len(train_X), len(val_X), len(test_X))

unique_labels = set(label for sent in train_y for label in sent)
label_list = sorted(list(unique_labels))
print(label_list)
tag2id = {tag: id for id, tag in enumerate(label_list)}
id2tag = {id: tag for tag, id in tag2id.items()}
if dataset_name == 'sofc_slot':
    id2tag[tag2id['B-experiment_evoking_word']] = 'O'
    id2tag[tag2id['I-experiment_evoking_word']] = 'O'
num_labels = len(label_list)

cnt = dict()
for sent in train_y:
    for label in sent:
        if label[0] in ['I', 'B']: tag = label[2:]
        else: continue
        if tag not in cnt: cnt[tag] = 1
        else: cnt[tag] += 1

eval_labels = sorted([l for l in cnt.keys() if l != 'experiment_evoking_word'])

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
    'model_max_length': 512
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)


def remove_zero_len_tokens(X, y):
    new_X, new_y = [], []
    for sent, labels in zip(X, y):
        new_sent, new_labels = [], []
        for token, label in zip(sent, labels):
            if len(tokenizer.tokenize(token)) == 0:
                assert dataset_name == 'matscholar'
                continue
            new_sent.append(token)
            new_labels.append(label)
        new_X.append(new_sent)
        new_y.append(new_labels)
    return new_X, new_y


train_X, train_y = remove_zero_len_tokens(train_X, train_y)
val_X, val_y = remove_zero_len_tokens(val_X, val_y)
test_X, test_y = remove_zero_len_tokens(test_X, test_y)

train_encodings = tokenizer(train_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
test_encodings = tokenizer(test_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


train_labels = encode_tags(train_y, train_encodings)
val_labels = encode_tags(val_y, val_encodings)
test_labels = encode_tags(test_y, test_encodings)

train_encodings.pop('offset_mapping')
val_encodings.pop('offset_mapping')
test_encodings.pop('offset_mapping')


class NER_Dataset(torch.utils.data.Dataset):
    def __init__(self, inp, labels):
        self.inp = inp
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NER_Dataset(train_encodings, train_labels)
val_dataset = NER_Dataset(val_encodings, val_labels)
test_dataset = NER_Dataset(test_encodings, test_labels)

config_kwargs = {
    'num_labels': num_labels,
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    preds, labs = [], []
    for pred, lab in zip(true_predictions, true_labels):
        preds.extend(pred)
        labs.extend(lab)
    assert(len(preds) == len(labs))
    labels_and_predictions = [" ".join([str(i), labs[i], preds[i]]) for i in range(len(labs))]
    counts = conlleval.evaluate(labels_and_predictions)
    scores = conlleval.get_scores(counts)
    results = {}
    macro_f1 = 0
    for k in eval_labels:
        if k in scores:
            results[k] = scores[k][-1]
        else:
            results[k] = 0.0
        macro_f1 += results[k]
    macro_f1 /= len(eval_labels)
    results['macro_f1'] = macro_f1 / 100
    results['micro_f1'] = conlleval.metrics(counts)[0].fscore
    return results


metric_for_best_model = 'macro_f1' if dataset_name[:4] == 'sofc' else 'micro_f1'
other_metric = 'micro_f1' if metric_for_best_model == 'macro_f1' else 'macro_f1'

best_lr = None
best_val = 0
best_val_acc_list = None
best_test_acc_list = None
best_val_oth_list = None
best_test_oth_list = None

if dataset_name == 'sofc':
    num_epochs = 20
elif dataset_name == 'sofc_slot':
    num_epochs = 40
elif dataset_name == 'matscholar':
    num_epochs = 15
else:
    raise NotImplementedError

arch = args.architecture if args.architecture != 'bert-bilstm-crf' else f'bert-bilstm-crf-{args.hidden_dim}'


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
            evaluation_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            save_total_limit=2,
            warmup_ratio=0.1,
            learning_rate=lr,
            seed=SEED
        )

        if args.architecture == 'bert':
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, from_tf=False, config=config,
                cache_dir=cache_dir, revision=model_revision, use_auth_token=None,
            )
        elif args.architecture == 'bert-crf':
            model = BERT_CRF(model_name, device, config, cache_dir)
        elif args.architecture == 'bert-bilstm-crf':
            model = BERT_BiLSTM_CRF(model_name, device, config, cache_dir, hidden_dim=args.hidden_dim)
        else:
            raise NotImplementedError
        model = model.to(device)
        
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
        val_acc.append(val_result['eval_' + metric_for_best_model])
        val_oth.append(val_result['eval_' + other_metric])

        test_result = trainer.evaluate(test_dataset)
        print(test_result)
        test_acc.append(test_result['eval_' + metric_for_best_model])
        test_oth.append(test_result['eval_' + other_metric])

        if preds_save_dir:
            val_preds = trainer.predict(val_dataset).predictions
            test_preds = trainer.predict(test_dataset).predictions

            for split, preds in zip(['val', 'test'], [val_preds, test_preds]):
                file_path = os.path.join(preds_save_dir, f'{split}_{args.model_name}_{arch}_{lr}_{SEED}.pkl')
                pickle.dump(preds, open(file_path, 'wb'))

    if np.mean(val_acc) > best_val:
        best_val = np.mean(val_acc)
        best_lr = lr
        best_val_acc_list = val_acc
        best_test_acc_list = test_acc
        best_val_oth_list = val_oth
        best_test_oth_list = test_oth


print(args.model_name, dataset_name, args.architecture)
print(f'best_lr: {best_lr}')
print(f'best_val: {best_val}')
print(f'best_val {metric_for_best_model}: {best_val_acc_list}')
print(f'best_test {metric_for_best_model}: {best_test_acc_list}')
print(f'best_val {other_metric}: {best_val_oth_list}')
print(f'best_test {other_metric}: {best_test_oth_list}')

if preds_save_dir:
    idxs = [f'Val {metric_for_best_model}', f'Test {metric_for_best_model}', f'Val {other_metric}', f'Test {other_metric}']
    res = pd.DataFrame([best_val_acc_list, best_test_acc_list, best_val_oth_list, best_test_oth_list], index=idxs)
    file_path = os.path.join(preds_save_dir, f'res_{args.model_name}_{arch}.pkl')
    pickle.dump(res, open(file_path, 'wb'))

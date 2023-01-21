import csv
import os
import re
import sys
sys.path.append('..')

from normalize_text import normalize


def get_sofc_data_split(split_name, data_dir, doc_info, slotting):
    assert split_name in ['train', 'dev', 'test']
    tokens, labels = [], []
    
    for file in sorted(os.listdir(os.path.join(data_dir, 'texts'))):
        
        doc_name = file[3:-4]
        if doc_info[doc_name] != split_name:
            continue
    
        text_file = open(os.path.join(data_dir, 'texts', file))
        text = text_file.read()
        text_file.close()

        sent_file = open(os.path.join(data_dir, 'annotations', 'sentences', doc_name + '.csv'))
        sent_ann = sent_file.read()
        sent_file.close()

        sentence_offsets = dict()
        for line in sent_ann.split('\n'):
            l = line.split()
            if len(l) == 0: continue
            assert(len(l) == 4)
            sentence_offsets[int(l[0])] = (int(l[2]), int(l[3]))

        token_file = open(os.path.join(data_dir, 'annotations', 'entity_types_and_slots', doc_name + '.csv'))
        token_ann = token_file.read()
        token_file.close()

        prev_sent_id = None

        for line in token_ann.split('\n'):
            l = line.split()
            if len(l) == 0: continue
            assert(len(l) == 6)
            sent_id = int(l[0])
            if prev_sent_id is None or sent_id != prev_sent_id:
                tokens.append([])
                labels.append([])
            prev_sent_id = sent_id
            s = sentence_offsets[sent_id][0] + int(l[2])
            e = sentence_offsets[sent_id][0] + int(l[3])
            tokens[-1].append(text[s:e])
            labels[-1].append(l[5] if slotting else l[4])
            if slotting and labels[-1][-1][2:] == 'interconnect_material':
                labels[-1][-1] = 'O'
    return tokens, labels


def modify_cross_val_data_split(doc_info, fold):
    # sort training document IDs alphabetically
    train_ids = sorted([docid for docid in doc_info if doc_info[docid] in set(['train', 'dev'])])
    fold -= 1
    for docid in train_ids:
        doc_info[docid] = 'train'
    for i in range(fold, len(train_ids), 5):
        docid = train_ids[i]
        doc_info[docid] = 'dev'
    return doc_info


def get_sofc_data(slotting, fold):
    data_dir = 'datasets/sofc-exp-corpus'

    doc_info = dict()
    with open(os.path.join(data_dir, 'SOFC-Exp-Metadata.csv'), encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        header = next(csvreader)
        for row in csvreader:
            docid = row[header.index('name')]
            doc_info[docid] = row[header.index('set')]
            assert(doc_info[docid] in ['train', 'dev', 'test'])
            
    if fold is not None:
        assert (fold >= 1 and fold <= 5)
        doc_info = modify_cross_val_data_split(doc_info, fold)
    
    train_X, train_y = get_sofc_data_split('train', data_dir, doc_info, slotting)
    val_X, val_y = get_sofc_data_split('dev', data_dir, doc_info, slotting)
    test_X, test_y = get_sofc_data_split('test', data_dir, doc_info, slotting)
    
    return train_X, train_y, val_X, val_y, test_X, test_y


def parse_file(f_name):
    f = open(f_name, 'r')
    data = re.split(r'\n\s*\n', f.read().strip())
    f.close()
    tokens, labels = [], []
    for sent in data:
        sent_tokens, sent_labels = [], []
        for line in sent.split('\n'):
            l = re.split(r' +', line)
            if len(l) != 2:
                sent_tokens = []
                break
            if len(l[0]) == 0: l[0] = ' '
            if len(l[1]) == 0: l[1] = 'O'
            sent_tokens.append(l[0])
            sent_labels.append(l[1])
        if len(sent_tokens) > 0:
            tokens.append(sent_tokens)
            labels.append(sent_labels)
    return tokens, labels


def get_matscholar_data():
    data_dir = 'datasets/NER_MATSCHOLAR/'
    train_X, train_y = parse_file(os.path.join(data_dir, 'train.txt'))
    val_X, val_y = parse_file(os.path.join(data_dir, 'dev.txt'))
    test_X, test_y = parse_file(os.path.join(data_dir, 'test.txt'))
    return train_X, train_y, val_X, val_y, test_X, test_y


def get_ner_data(data_name, fold=None, norm=False):
    data_name = data_name.lower()
    if data_name == 'sofc':
        data = get_sofc_data(slotting=False, fold=fold)
    elif data_name == 'sofc_slot':
        data = get_sofc_data(slotting=True, fold=fold)
    elif data_name == 'matscholar':
        data =  get_matscholar_data()
    else:
        raise NotImplementedError
    
    if norm == False:
        return data
    norm_data = []
    for split in data:
        norm_split = []
        for s in split:
            norm_split.append(normalize('\n'.join(s)).split('\n'))
        norm_data.append(norm_split)
    return norm_data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inp, labels):
        self.inp = inp
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
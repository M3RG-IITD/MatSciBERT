import os
from pathlib import Path

data_dir = '../datasets/annotated-materials-syntheses'


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

    all_data = ''
    k = 0
    for x_, y_ in zip(X, y):
        text, (s1, e1), (s2, e2) = x_[:3]
        if s1 < s2:
            t = text[:s1] + '<e1>' + text[s1:e1] + '</e1>' + text[e1:s2] + '<e2>' + text[s2:e2] + '</e2>' + text[e2:]
        else:
            t = text[:s2] + '<e2>' + text[s2:e2] + '</e2>' + text[e2:s1] + '<e1>' + text[s1:e1] + '</e1>' + text[e1:]
        all_data += f'{k}\t ' + t + ' \n' + y_ + '\n\n\n'
        k += 1
    
    return all_data


splits = ['train', 'dev', 'test']
for split in splits:
    data = get_data_of_split(split)
    f = open(os.path.join(data_dir, f'{split}.txt'), 'w')
    f.write(data)
    f.close()

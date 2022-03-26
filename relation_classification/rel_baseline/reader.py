import os
import pdb
import sys
import json
import random

sys.path.append(os.path.abspath('.'))


def fwrite(new_doc, path, mode='w', no_overwrite=False):
    if not path:
        print("[Info] Path does not exist in fwrite():", str(path))
        return
    if no_overwrite and os.path.isfile(path):
        print("[Error] pls choose whether to continue, as file already exists:",
              path)
        import pdb
        pdb.set_trace()
        return
    with open(path, mode) as f:
        f.write(new_doc)


class NLP:
    def __init__(self):
        import spacy

        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split())
        if lower: text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences


def load_data(path):
    ENT_1_START = '<e1>'
    ENT_1_END = '</e1>'
    ENT_2_START = '<e2>'
    ENT_2_END = '</e2>'

    nlp = NLP()
    data = []
    with open(path) as f:
        lines = [line.strip() for line in f]
    for idx in range(0, len(lines), 4):
        id = int(lines[idx].split("\t")[0])
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        # pdb.set_trace()
        sentence = sentence.strip()

        sentence = sentence.replace(ENT_1_START, ' ENT_1_START ')
        sentence = sentence.replace(ENT_1_END, ' ENT_1_END ')
        sentence = sentence.replace(ENT_2_START, ' ENT_2_START ')
        sentence = sentence.replace(ENT_2_END, ' ENT_2_END ')

        sentence = nlp.word_tokenize(sentence)

        ent1 = sentence.split(' ENT_1_START ')[-1].split(' ENT_1_END ')[0]
        ent2 = sentence.split(' ENT_2_START ')[-1].split(' ENT_2_END ')[0]


        data.append({
            'label': relation,
            'sentence': sentence,
            'ent1': ent1,
            'ent2': ent2,
            'id': id,
        })

    return data


def save_to_json(data, file):
    writeout = json.dumps(data, indent=4)
    fwrite(writeout, file)
    print('[Info] Saving {} data to {}'.format(len(data), file))


def main():

    data = {}
    data_dir = '../datasets/annotated-materials-syntheses'

    for split in ['train', 'valid', 'test']:
        f_name = split if split != 'valid' else 'dev'
        raw_fname = os.path.join(data_dir, f'{f_name}.txt')
        data[split] = load_data(raw_fname)

    for key, value in data.items():
        json_fname = os.path.join(data_dir, f'{key}.json')
        save_to_json(value, json_fname)


if __name__ == "__main__":
    main()


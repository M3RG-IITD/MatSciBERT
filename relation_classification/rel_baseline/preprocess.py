import os
import csv
import json
import configargparse


def json2csv(json_fname, csv_fname, args):
    with open(json_fname) as f:
        data = json.load(f)
    csv_data = []
    for line in data:
        sentence = line['sentence']
        sentence = ' '.join(sentence.split()[:args.sent_max_len])
        if args.lower: sentence = sentence.lower()

        csv_line = {
            'tgt': line['label'],
            'input': sentence,
            'show_inp': sentence,
            'ent1': line['ent1'],
            'ent2': line['ent2'],
            'id': line['id'],
        }
        csv_data += [csv_line]
    with open(csv_fname, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_line.keys())
        writer.writeheader()
        writer.writerows(csv_data)
    print('[Info] Writing {} data to {}'.format(len(csv_data), csv_fname))


def get_args():
    parser = configargparse.ArgumentParser(
        description='Options for preprocessing')
    parser.add_argument('-lower', action='store_true', default=False,
                        help='whether to keep the uppercase')
    parser.add_argument('-sent_max_len', default=550, type=int,
                        help='the maximum number of words allowed in a sentence')
    parser.add_argument('-data_dir', default='../datasets/annotated-materials-syntheses', type=str,
                        help='path to load data from')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_dir = args.data_dir

    for typ in 'train valid test'.split():
        json_fname = os.path.join(data_dir, f'{typ}.json')
        csv_fname = os.path.join(data_dir, f'{typ}.csv')
        json2csv(json_fname, csv_fname, args)


if __name__ == '__main__':
    main()

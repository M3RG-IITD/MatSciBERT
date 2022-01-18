import json
import torch
import torchtext

from torchtext.data import Field, RawField, TabularDataset, \
    BucketIterator, Iterator
from torchtext.vocab import Vectors, GloVe
from utils import show_time, fwrite


class Dataset:
    def __init__(self, proc_id=0, data_dir='tmp/', train_fname='train.csv',
                 preprocessed=True, lower=True,
                 vocab_max_size=100000, emb_dim=100,
                 save_vocab_fname='vocab.json', verbose=True, ):
        self.verbose = verbose and (proc_id == 0)
        tokenize = lambda x: x.split() if preprocessed else 'spacy'

        INPUT = Field(sequential=True, batch_first=True, tokenize=tokenize,
                      lower=lower,
                      # include_lengths=True,
                      )
        # TGT = Field(sequential=False, dtype=torch.long, batch_first=True,
        #             use_vocab=False)
        TGT = Field(sequential=True, batch_first=True)
        SHOW_INP = RawField()
        fields = [
            ('tgt', TGT),
            ('input', INPUT),
            ('show_inp', SHOW_INP), ]

        if self.verbose:
            show_time("[Info] Start building TabularDataset from: {}{}"
                      .format(data_dir, 'train.csv'))
        datasets = TabularDataset.splits(
            fields=fields,
            path=data_dir,
            format=train_fname.rsplit('.')[-1],
            train=train_fname,
            validation=train_fname.replace('train', 'valid'),
            test=train_fname.replace('train', 'test'),
            skip_header=True,
        )
        INPUT.build_vocab(*datasets, max_size=vocab_max_size,
                          vectors=GloVe(name='6B', dim=emb_dim),
                          unk_init=torch.Tensor.normal_, )
        # load_vocab(hard_dosk) like opennmt
        # emb_dim = {50, 100}
        # Elmo
        TGT.build_vocab(*datasets)

        self.INPUT = INPUT
        self.TGT = TGT
        self.train_ds, self.valid_ds, self.test_ds = datasets

        if save_vocab_fname and self.verbose:
            writeout = {
                'tgt_vocab': {
                    'itos': TGT.vocab.itos, 'stoi': TGT.vocab.stoi,
                },
                'input_vocab': {
                    'itos': INPUT.vocab.itos, 'stoi': INPUT.vocab.stoi,
                },
            }
            fwrite(json.dumps(writeout, indent=4), save_vocab_fname)

        if self.verbose:
            msg = "[Info] Finished building vocab: {} INPUT, {} TGT" \
                .format(len(INPUT.vocab), len(TGT.vocab))
            show_time(msg)

    def get_dataloader(self, proc_id=0, n_gpus=1, device=torch.device('cpu'),
                       batch_size=64):
        def _distribute_dataset(dataset):
            n = len(dataset)
            part = dataset[n * proc_id // n_gpus: n * (proc_id + 1) // n_gpus]
            return torchtext.data.Dataset(part, dataset.fields)

        train_ds = _distribute_dataset(self.train_ds)
        self.verbose = self.verbose and (proc_id == 0)
        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, self.valid_ds),
            batch_sizes=(batch_size, batch_size),
            sort_within_batch=True,
            sort_key=lambda x: len(x.input),
            device=device,
            repeat=False,
        )

        test_iter = Iterator(
            self.test_ds,
            batch_size=1,
            sort=False,
            sort_within_batch=False,
            device=device,
            repeat=False,
        )
        train_dl = BatchWrapper(train_iter)
        valid_dl = BatchWrapper(valid_iter)
        test_dl = BatchWrapper(test_iter)
        return train_dl, valid_dl, test_dl


class BatchWrapper:
    def __init__(self, iterator):
        self.iterator = iterator

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield batch


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


if __name__ == '__main__':
    from tqdm import tqdm

    file_dir = "~/proj/1908_prac_toxic/data/yelp/"
    dataset = Dataset(data_dir=file_dir)
    train_dl, valid_dl, test_dl = dataset.get_dataloader()
    show_time('[Info] Begin iterating 10 epochs')
    for epoch in range(10):
        for batch in tqdm(train_dl):
            pass
            # inpect padding num distribution
            # use `pack_padded_sequence`
    show_time('[Info] Finished loading')


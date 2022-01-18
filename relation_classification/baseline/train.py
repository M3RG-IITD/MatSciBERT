from __future__ import division, print_function
from tqdm import tqdm
import torch

from utils import shell, init_weights, set_seed

from get_args import setup, model_setup, clean_up
from dataloader import Dataset
from model import LSTMClassifier
from evaluate import Validator, Predictor


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train(proc_id, n_gpus, model=None, train_dl=None, validator=None,
          tester=None, epochs=20, lr=0.001, log_every_n_examples=1,
          weight_decay=0):
    # opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                       lr=lr, momentum=0.9)
    opt = torch.optim.Adadelta(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1.0, rho=0.9,
        eps=1e-6, weight_decay=weight_decay)

    for epoch in range(epochs):
        if epoch - validator.best_epoch > 10:
            return

        model.train()
        pbar = tqdm(train_dl) if proc_id == 0 else train_dl
        total_loss = 0
        n_correct = 0
        cnt = 0
        for batch in pbar:
            batch_size = len(batch.tgt)

            if proc_id == 0 and cnt % log_every_n_examples < batch_size:
                pbar.set_description('E{:02d}, loss:{:.4f}, acc:{:.4f}, lr:{}'
                                     .format(epoch,
                                             total_loss / cnt if cnt else 0,
                                             n_correct / cnt if cnt else 0,
                                             opt.param_groups[0]['lr']))
                pbar.refresh()

            loss, acc = model.loss_n_acc(batch.input, batch.tgt)
            total_loss += loss.item() * batch_size
            cnt += batch_size
            n_correct += acc

            opt.zero_grad()
            loss.backward()
            clip_gradient(model, 1)
            opt.step()

        if n_gpus > 1: torch.distributed.barrier()

        model.eval()
        validator.evaluate(model, epoch)
        # tester.evaluate(model, epoch)
        if proc_id == 0:
            summ = {
                'Eval': '(e{:02d},train)'.format(epoch),
                'loss': total_loss / cnt,
                'acc': n_correct / cnt,
            }
            validator.write_summary(summ=summ)
            validator.write_summary(epoch=epoch)

            # tester.write_summary(epoch)


def bookkeep(predictor, validator, tester, args, INPUT_field, label_list):
    validator.final_evaluate(predictor.model, label_list, 'valid')
    tester.final_evaluate(predictor.model, label_list, 'test')

    predictor.pred_sent(INPUT_field)

    save_model_fname = validator.save_model_fname + '.e{:02d}.loss{:.4f}.torch'.format(
        validator.best_epoch, validator.best_loss)
    cmd = 'cp {} {}'.format(validator.save_model_fname, save_model_fname)
    shell(cmd)

    clean_up(args)


def run(proc_id, n_gpus, devices, args):
    set_seed(args.seed)
    dev_id = devices[proc_id]

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port=args.tcp_port)
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=dev_id)
    device = torch.device(dev_id) if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    dataset = Dataset(proc_id=proc_id, data_dir=args.save_dir,
                      train_fname=args.train_fname,
                      preprocessed=args.preprocessed, lower=args.lower,
                      vocab_max_size=args.vocab_max_size, emb_dim=args.emb_dim,
                      save_vocab_fname=args.save_vocab_fname, verbose=True, )
    label_list = dataset.TGT.vocab.itos.copy()
    label_list.remove("<unk>")
    label_list.remove("<pad>")
    train_dl, valid_dl, test_dl = \
        dataset.get_dataloader(proc_id=proc_id, n_gpus=n_gpus, device=device,
                               batch_size=args.batch_size)

    validator = Validator(dataloader=valid_dl, save_dir=args.save_dir,
                          save_log_fname=args.save_log_fname,
                          save_model_fname=args.save_model_fname,
                          valid_or_test='valid',
                          vocab_itos=dataset.INPUT.vocab.itos,
                          label_itos=dataset.TGT.vocab.itos)
    tester = Validator(dataloader=test_dl, save_log_fname=args.save_log_fname,
                       save_dir=args.save_dir, valid_or_test='test',
                       vocab_itos=dataset.INPUT.vocab.itos,
                       label_itos=dataset.TGT.vocab.itos)
    predictor = Predictor(args.save_vocab_fname)

    if args.load_model:
        predictor.use_pretrained_model(args.load_model, device=device)
        import pdb;
        pdb.set_trace()

        predictor.pred_sent(dataset.INPUT)
        validator.final_evaluate(predictor.model, label_list, 'valid')
        tester.final_evaluate(predictor.model, label_list, 'test')

        return

    model = LSTMClassifier(emb_vectors=dataset.INPUT.vocab.vectors,
                           emb_dropout=args.emb_dropout,
                           lstm_dim=args.lstm_dim,
                           lstm_n_layer=args.lstm_n_layer,
                           lstm_dropout=args.lstm_dropout,
                           lstm_combine=args.lstm_combine,
                           linear_dropout=args.linear_dropout,
                           n_linear=args.n_linear,
                           n_classes=len(dataset.TGT.vocab))
    if args.init_xavier: model.apply(init_weights)
    model = model.to(device)
    args = model_setup(proc_id, model, args)

    train(proc_id, n_gpus, model=model, train_dl=train_dl,
          validator=validator, tester=tester, epochs=args.epochs, lr=args.lr,
          weight_decay=args.weight_decay)

    if proc_id == 0:
        predictor.use_pretrained_model(args.save_model_fname, device=device)
        bookkeep(predictor, validator, tester, args, dataset.INPUT, label_list)


def main():
    args = setup()

    n_gpus = args.n_gpus
    devices = range(n_gpus)

    if n_gpus == 1:
        run(0, n_gpus, devices, args)
    else:
        mp = torch.multiprocessing
        mp.spawn(run, args=(n_gpus, devices, args), nprocs=n_gpus)


if __name__ == '__main__':
    main()

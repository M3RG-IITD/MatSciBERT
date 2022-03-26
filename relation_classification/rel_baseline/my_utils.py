import torch
import random
import numpy as np
# from sklearn.metrics import f1_score


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return int(correct.sum()), int(y.shape[0])


# def f1_score(preds,y):
#     from sklearn.metrics import f1_score
#     y_pred = list(preds.argmax(dim = 1, keepdim = True).squeeze(1).cpu().detach().long().numpy()) # get the index of the max probability
#     y = list(y.cpu().detach().long().numpy())
#     score = f1_score(y, y_pred)#, average='macro')
#     return score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def set_all_seeds_to(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def get_model_path(args):
    if(args.model_path == ''):
        path = str(args.wiki) + '_' + str(args.lr) + '_' + str(args.batch_size)
        if args.debug == 1:
            path += '_debug'
        if args.forget_bias != 1:
            path += '_fb_'+str(args.forget_bias)
        if args.weight_decay != 0:
            path += '_decay_' + str(args.weight_decay)   
        if args.drop_connect != 0:
            path += '_dc_' + str(args.drop_connect)   
        if args.gradients == True:
            path += '_grad'
        if args.glove == False:
            path += '_noGlove'
        if args.freeze_embedding == 1:
            path += '_freeze'
        if args.use_bert:
            path += '_bert'
        if args.amsgrad:
            path += '_amsgrad'
        if args.nesterov:
            path += '_nesterov'
        return path
    else:
        return args.model_path


def get_all_logs(args, model_dir):
    logf = open(model_dir+'/log.txt','a')
    if args.mode != "train":
        train_acc_f = open(model_dir+'/train.txt','a')
        valid_acc_f = open(model_dir+'/dev.txt','a')
        test_acc_f = open(model_dir+'/test.txt','a')
    elif args.mode == "train":
        train_acc_f = open(model_dir+'/train.txt','w')
        valid_acc_f = open(model_dir+'/dev.txt','w')
        test_acc_f = open(model_dir+'/test.txt','w')    
    ratios_f = open(model_dir + '/ratios.txt','w') if args.gradients else None
    return logf, train_acc_f, valid_acc_f, test_acc_f, ratios_f


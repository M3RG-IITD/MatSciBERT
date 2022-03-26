import argparse
from distutils import util

def parse_args():
    parser = argparse.ArgumentParser()

    ## Basics
    parser.add_argument('--config_file', help="Configuration file containing parameters",
                    type=str)
    parser.add_argument('--pool', type=str, required=True) # max, last or last1, mean
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='') # optional, in case of test, if not standard format
    parser.add_argument('--mode', type=str, required=True) # train, test, resume, curriculum
    parser.add_argument('--debug', type=int, default=0) # 1 for debug mode    
    parser.add_argument('--task', type=str, default='IMDB') # IMDB, ...
    # parser.add_argument('--data_size', type=str, default='10K') # 1K, 2K, 5K, 10K, 25K
    parser.add_argument('--customlstm', type=int, default=1, choices=[0,1]) 
    

    parser.add_argument('--log', type=int, default=0, choices=[0,1]) 

    ## Exps
    parser.add_argument('--ood', type=int, default=0) # Out of distribution
    parser.add_argument('--vec', type=int, default=0) # 0 or 1 --> for random vectors
    parser.add_argument('--wiki', type=str, default="none") # left, mid, right or none --> for wikipedia words exp --> wiki_left means original is on left
    parser.add_argument('--gradients', default=False, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--ratios', default=False, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--initial', default=False, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--gates', default=False, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--NWI', type=int, default=0) # For Positional variation
    parser.add_argument('--explain', type=int, default=0) # For Explanation in case of attention, max

    ## Hyper-parameters - amsgrad
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--bidirectional', default=True, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--glove', default=True, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--use_embedding', default=True, type=lambda x:bool(util.strtobool(x))) #For MNIST keep False
    parser.add_argument('--use_bert', default=False, type=lambda x:bool(util.strtobool(x))) #For MNIST keep False
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--amsgrad', default=False, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--nesterov', default=False, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--momentum', type=float, default = 0.9)
    parser.add_argument('--clip', type=float, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--cap_vocab', default=True, type=lambda x:bool(util.strtobool(x)))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--forget_bias', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--drop_connect', type=float, default=0)
    parser.add_argument('--penalty_power', type=int, default=4)
    parser.add_argument('--teach_small', type=int, default=0)
    parser.add_argument('--drop_strategy', type=int, default=0)
    parser.add_argument('--freeze_embedding', type=int, default=0)
    parser.add_argument('--req_ex', type=int, default=500)
    return parser

def add_config(args):
    data = yaml.load(open(args.config_file,'r'))

    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    return args


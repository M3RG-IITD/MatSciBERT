import os, sys
from tqdm import tqdm

import torch

from transformers import (
    AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


model_revision = 'main'
root_dir = '/home/maths/dual/mt6170499/scratch'
model_name = 'allenai/scibert_scivocab_uncased'
cache_dir = os.path.join(root_dir, '.cache')
output_dir = os.path.join(root_dir, 'wwm/output_dataset_final/scibert_uncased')


SEED = 42
set_seed(SEED)

config_kwargs = {
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

max_seq_length = 512

start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')
print(start_tok, sep_tok, pad_tok)


def full_sent_tokenize(file_name):
    f = open(file_name, 'r')
    sents = f.read().strip().split('\n')
    f.close()
    
    tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in tqdm(sents)]
    for s in tok_sents:
        s.pop(0)
    
    res = [[]]
    l_curr = 0
    
    for s in tok_sents:
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
            l_curr = len(res[-1])
    
    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length
        
    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))
    
    return {'input_ids': res, 'attention_mask': attention_mask}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        return item

    def __len__(self):
        return len(self.inp['input_ids'])


train_file = os.path.join(root_dir, 'wwm/corpus_final/train_150000_corpus_norm.txt')
eval_file = os.path.join(root_dir, 'wwm/corpus_final/val_150000_corpus_norm.txt')
train_dataset = MyDataset(full_sent_tokenize(train_file))
eval_dataset = MyDataset(full_sent_tokenize(eval_file))

print(len(train_dataset), len(eval_dataset))


model = BertForMaskedLM.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=None,
)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    warmup_ratio=0.048,
    learning_rate=1e-4,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    num_train_epochs=30,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

resume = None if len(os.listdir(output_dir)) == 0 else True
print('Resume from checkpoint: ', resume)
train_res = trainer.train(resume_from_checkpoint=True)
print(train_res)

train_output = trainer.evaluate(train_dataset)
eval_output = trainer.evaluate()

print(train_output)
print(eval_output)

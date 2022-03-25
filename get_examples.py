import json
import csv
import torch
import transformers
import numpy as np
import tqdm
import os

from cobert import match, aggregate, fuse, dot_product_loss, train_epoch, evaluate_epoch
from dataset import PersonaChatTorchDataset, clf, tokenize
from util import logger
                        
with open('config.json', 'r') as config:
    config  = json.loads(config.read())

save_model_path = config['save_model_path']

gradient_accumulation_steps = config['gradient_accumulation_steps']

matching_method = config['matching_method']

lr =  config['lr'] 
warmup_steps = config['warmup_steps']
test_mode = config['test_mode']
has_persona = config['has_persona']

context_len = config['context_len']
responce_len = config['responce_len']
persona_len = config['persona_len']
train_batch_size = config['train_batch_size']
val_batch_size = config['val_batch_size']
epochs = config['epochs']
split = config['split']

no_decay = ["bias", "LayerNorm.weight"]
fp16 = False
amp = None
weight_decay = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bert_path = config['model_path']
proc_data = config['proc_data_path']
apply_interaction = config['apply_interaction']
aggregation_method = config['aggregation_method']
padding_side = config['padding_side']

bert_config = transformers.BertConfig.from_pretrained(bert_path)
bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, padding_side=padding_side)

data = PersonaChatTorchDataset(proc_data)
split = len(data)//config['split']
train, val = torch.utils.data.random_split(data, [len(data)-split, split])
train = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                 tokenizer=bert_tokenizer, 
                                                 context_len=context_len, 
                                                 responce_len=responce_len, 
                                                 persona_len=persona_len,
                                                 type='bert',
                                                 persona_use='split'))
val = torch.utils.data.DataLoader(val, batch_size=val_batch_size,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                 tokenizer=bert_tokenizer, 
                                                 context_len=context_len, 
                                                 responce_len=responce_len, 
                                                 persona_len=persona_len,
                                                 type='bert',
                                                 persona_use='concat'))
print('\ntrain:', len(train), 'val:', len(val))
t_total = len(train) // gradient_accumulation_steps * train_batch_size

for x, y in train:
  context = x['context']
  responce = x['responce']
  persona = x['persona']
  print(context['input_ids'])
  context_text = bert_tokenizer.decode(context['input_ids'][0])
  persona_text = bert_tokenizer.decode(persona['input_ids'][0])
  print(context_text)
  print(persona_text)
  break

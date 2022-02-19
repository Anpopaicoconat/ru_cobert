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

model_path = config['model_path']
padding_side = config['padding_side']
save_model_path = config['save_model_path']

gradient_accumulation_steps = config['gradient_accumulation_steps'] 
apply_interaction = config['apply_interaction'] 
matching_method = config['matching_method']
aggregation_method = config['aggregation_method']
lr =  config['lr'] 
warmup_steps = config['warmup_steps']
test_mode = config['test_mode']
has_persona = config['has_persona']
log_path = config['log_path']

context_len = config['context_len']
responce_len = config['responce_len']
persona_len = config['persona_len']
batch_size = config['batch_size']
epochs = config['epochs']
split = config['split']

no_decay = ["bias", "LayerNorm.weight"]
fp16 = False
amp = None
optimizers = []
schedulers = []
weight_decay = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_path, padding_side=padding_side, local_files_only=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_tokens({'end_of_masage_token': '[EOM]', 'end_of_masage_token': '[EOPS]'}, special_tokens=True)
tokenizer.end_of_masage_token = '[EOM]'
tokenizer.end_of_persona_sentence_token = '[EOPS]'
model = transformers.GPT2Model.from_pretrained(model_path, local_files_only=True)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)
models = [model]

data = PersonaChatTorchDataset(config['proc_data_path'])
split = len(data)//config['split']
train, val = torch.utils.data.random_split(data, [len(data)-split, split])
train = torch.utils.data.DataLoader(train, batch_size=32,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                 tokenizer=tokenizer, 
                                                 context_len=context_len, 
                                                 responce_len=responce_len, 
                                                 persona_len=persona_len))
val = torch.utils.data.DataLoader(val, batch_size=32,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                 tokenizer=tokenizer, 
                                                 context_len=context_len, 
                                                 responce_len=responce_len, 
                                                 persona_len=persona_len))
print(len(train), len(val))

t_total = len(train) // gradient_accumulation_steps * 1

for i, model in enumerate(models):
    UNFREEZE_LAST_N = 8
    for param in list(model.parameters())[:-1]:
        param.requires_grad = False
    for i, m in enumerate(model.h):        
        #Only un-freeze the last n transformer blocks
        if i+1 > len(model.h) - UNFREEZE_LAST_N:
            print("un-freeze block number {} ".format(i+1))
            for parameter in m.parameters():
                parameter.requires_grad = True 

    for parameter in model.ln_f.parameters():        
        parameter.requires_grad = True

    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr = lr, # default is 5e-5
                                   eps = 1e-8 # default is 1e-8.
                                   )

    optimizers.append(optimizer)
    
    if not test_mode:
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        schedulers.append(scheduler)

epoch_train_losses = []
epoch_valid_losses = []
epoch_valid_accs = []
epoch_valid_recalls = []
epoch_valid_MRRs = []
best_model_statedict = {}

with open(log_path, 'w') as log:
    writer = csv.DictWriter(log, fieldnames=['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'valid_recall', 'valid_MRR'], delimiter=',', quotechar='"')
    writer.writeheader()
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        # training
        for model in models:
            model.train()

        train = tqdm.tqdm(train, desc="Iteration")
        train_loss, (train_acc, _, _) = train_epoch(data_iter=train, 
                                                    models=models, has_persona=has_persona, optimizers=optimizers, 
                                                    schedulers=schedulers, 
                                                    gradient_accumulation_steps=gradient_accumulation_steps, 
                                                    device=device, fp16=fp16, 
                                                    amp=amp, apply_interaction=apply_interaction, 
                                                    matching_method=matching_method, 
                                                    aggregation_method=aggregation_method)
        epoch_train_losses.append(train_loss)

        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm.tqdm(val, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = evaluate_epoch(data_iter=val, models=models,
                                                                            has_persona=has_persona,
                                                                            gradient_accumulation_steps=gradient_accumulation_steps, 
                                                                            device=device, epoch=epoch, apply_interaction=apply_interaction, 
                                                                            matching_method=matching_method, aggregation_method=aggregation_method)

        print("Epoch {0}: train loss: {1:.4f}, valid loss: {2:.4f}, train_acc: {3:.4f}, valid acc: {4:.4f}, valid recall: {5}, valid_MRR: {6:.4f}"
            .format(epoch+1, train_loss, valid_loss, train_acc, valid_acc, valid_recall, valid_MRR))
        writer.writerow({'epoch':epoch+1, 'train_loss':train_loss, 'valid_loss': valid_loss, 'train_acc':train_acc, 'valid_acc':valid_acc, 'valid_recall':valid_recall, 'valid_MRR':valid_MRR})
        epoch_valid_losses.append(valid_loss)
        epoch_valid_accs.append(valid_acc)
        epoch_valid_recalls.append(valid_recall)
        epoch_valid_MRRs.append(valid_MRR)

        if save_model_path != "":
            if epoch == 0:
                for k, v in models[0].state_dict().items():
                    best_model_statedict[k] = v.cpu()
            else:
                if epoch_valid_recalls[-1][0] == max([recall1 for recall1, _, _ in epoch_valid_recalls]):
                    for k, v in models[0].state_dict().items():
                        best_model_statedict[k] = v.cpu()

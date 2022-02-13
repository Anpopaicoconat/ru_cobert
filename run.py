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

proc_data_path_list = ["data/personachat/enpersonachat.txt", "data/TlkPersonaChatRus/tolokapersonachat.txt"]
bert_path_list = ["models/enbert", "models/rubert"]

for proc_data, bert_path in zip(proc_data_path_list, bert_path_list):
    if proc_data == "data/personachat/enpersonachat.txt":
        continue
    for apply_interaction in range(1, 2):
        for aggregation_method in ['max', 'mean', 'cls']: #'max', 'mean', 'meanmax', 
            for padding_side in ['left', 'right']:
                log_path = bert_path.split('/')[-1] + '_' + proc_data.split('/')[-1].split('.')[0] + '_interaction' + str(apply_interaction) \
                + '_' + aggregation_method + '_' + padding_side + '.csv'
                if log_path in tuple(os.walk('logs/'))[0][2]:
                    print('skeep', log_path)
                    continue
                else:
                    log_path = 'logs/'+log_path
                print(log_path)
                bert_config = transformers.BertConfig.from_pretrained(bert_path)
                bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, padding_side=padding_side)
                bert_model = transformers.BertModel(bert_config).from_pretrained(bert_path)
                models = [bert_model]
                data = PersonaChatTorchDataset(proc_data)
                split = len(data)//config['split']
                train, val = torch.utils.data.random_split(data, [len(data)-split, split])
                train = torch.utils.data.DataLoader(train, batch_size=32,
                                        shuffle=True, num_workers=0, 
                                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                                 tokenizer=bert_tokenizer, 
                                                                 context_len=context_len, 
                                                                 responce_len=responce_len, 
                                                                 persona_len=persona_len))
                val = torch.utils.data.DataLoader(val, batch_size=32,
                                        shuffle=True, num_workers=0, 
                                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                                 tokenizer=bert_tokenizer, 
                                                                 context_len=context_len, 
                                                                 responce_len=responce_len, 
                                                                 persona_len=persona_len))
                print(len(train), len(val))


                t_total = len(train) // gradient_accumulation_steps * 1
                for i, model in enumerate(models):
                    optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": weight_decay,
                        },
                        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                    ]
                    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)


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

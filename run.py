import json
import csv
import torch
import transformers
import numpy as np
import tqdm

from cobert import match, aggregate, fuse, dot_product_loss, train_epoch, evaluate_epoch
from dataset import PersonaChatTorchDataset, clf, tokenize

with open('config.json', 'r') as config:
    config  = json.loads(config.read())

bert_config = transformers.BertConfig.from_pretrained(config['bert_path'])
bert_tokenizer = transformers.BertTokenizer.from_pretrained(config['bert_path'], padding_side='left')
bert_model = transformers.BertModel(bert_config).from_pretrained(config['bert_path'])
models = [bert_model]
save_model_path = config['save_model_path']

gradient_accumulation_steps = config['gradient_accumulation_steps'] 
apply_interaction = config['apply_interaction'] 
matching_method = config['apply_interaction']
aggregation_method = config['aggregation_method']
lr =  config['lr'] 
warmup_steps = config['warmup_steps']
test_mode = config['test_mode']

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
data = PersonaChatTorchDataset(config['proc_data_path'])
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

# train loop
for epoch in range(epochs):
        print("Epoch", epoch+1)
        # training
        for model in models:
            model.train()
            
        train = tqdm.tqdm(train, desc="Iteration")
        train_loss, (train_acc, _, _) = train_epoch(data_iter=train, 
                                                    models=models, optimizers=optimizers, 
                                                    schedulers=schedulers, 
                                                    gradient_accumulation_steps=gradient_accumulation_steps, 
                                                    device=device, fp16=fp16, 
                                                    amp=amp, apply_interaction=apply_interaction, 
                                                    matching_method=matching_method, 
                                                    aggregation_method=aggregation_method)
        epoch_train_losses.append(train_loss)

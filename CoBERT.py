import json
import csv
import torch
import transformers
import numpy as np

data_load_config = {
    "raw_data_path": "/content/drive/MyDrive/stagirovka/TlkPersonaChatRus/dialogues.tsv",
    "proc_data_path": "/content/drive/MyDrive/stagirovka/TlkPersonaChatRus/tolokapersonachat.txt",
    "rubert_path": "/content/drive/MyDrive/stagirovka/models/rubert"
    "gradient_accumulation_steps": 4
    "apply_interaction": True
    "matching_method": "CoBERT"
    "aggregation_method": "max"
    "lr": 2e-5
    "warmup_steps": 0
    "test_mode": False
}

###

bert_config = transformers.BertConfig.from_pretrained(data_load_config['rubert_path'])
bert_tokenizer = transformers.BertTokenizer.from_pretrained(data_load_config['rubert_path'])
bert_model = transformers.BertModel(bert_config).from_pretrained(data_load_config['rubert_path'])
models = [bert_model]

gradient_accumulation_steps = data_load_config['gradient_accumulation_steps'] 
apply_interaction = data_load_config['apply_interaction'] 
matching_method = data_load_config['apply_interaction']
aggregation_method = data_load_config['aggregation_method']
lr =  data_load_config['lr'] 
warmup_steps = data_load_config['warmup_steps']
test_mode = data_load_config['test_mode']

no_decay = ["bias", "LayerNorm.weight"]
fp16 = False
amp = None
optimizers = []
schedulers = []
weight_decay = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

###

def get_dialog(inp, mod):
    if inp[0]=='"':
        inp = inp[1:]

    inp = inp.replace('<br />', ' ').replace('\r', ' ').replace('\n', ' ').split('</span> ')
    out = [inp[0]]
    for i in inp[1:]:
        if i[:42] == out[-1][:42]:
            out[-1] = out[-1] + '. ' + i[42:]
        elif len(i)>25:
            out.append(i)
    return out
def bild_data():
    with open(data_load_config['raw_data_path'], 'r', encoding='utf-8') as data:
        data = csv.reader(data, delimiter='\t')
        with open(data_load_config['proc_data_path'], 'w', encoding='utf-8') as file:
            for i, conv in enumerate(data):
                if i == 0:
                    continue
                p1 = conv[0].replace('<span class=participant_1>', '')
                p2 = conv[1].replace('<span class=participant_2>', '')
                dialog = get_dialog(conv[2][:-1], 'join')
                for i in range(1, len(dialog)):
                    context = [p[42:] for p in dialog[:i]]
                    try:
                        persona, responce = dialog[i].replace('<span class=participant_1>Пользователь 1: ', p1+'[-sep-]')\
                        .replace('<span class=participant_2>Пользователь 2: ', p2+'[-sep-]').split('[-sep-]')
                    except BaseException:
                        print(conv)
                        C='a'
                        break

                    persona = persona.replace('</span>', '').split('<br />')[:-1]
                    file.write(json.dumps({'context':context, 'responce':responce, 'persona':persona, 'label':1})+'\n')

#bild_data()

###

def load_toloka():
    with open(data_load_config['proc_data_path'], 'r', encoding='utf-8') as data:
        for line in data:
            yield json.loads(line)

def tokenize_ru(inp, tokenizer=False, max_len=32, join_token=False):
    pad_id = bert_tokenizer.pad_token_id
    cls_id = bert_tokenizer.cls_token_id

    if join_token:
        out=[]
        for x in inp:
            out.append(join_token.join(x))
    else:
        out = inp 
    out = tokenizer(out, padding=True, truncation=False, return_tensors="pt")
    for k in out:
        ad_size = max_len-out[k].shape[1]
        if ad_size > 0:
            pad_mat = torch.zeros((out[k].shape[0], ad_size))
            out[k] = torch.cat((pad_mat, out[k]), dim=1)
        out[k] = out[k][:, -max_len:]
        cls_padder = torch.ones_like(out[k][:,:1])*cls_id
        out[k][:,:1] = torch.where((out[k][:,:1]!=pad_id), cls_padder, out[k][:,:1])
        out[k] = out[k].type(torch.IntTensor)
    return out

class TolokaLazyDataset():
    def __init__(self, path, tokenizer_func=False, tokenizer=False, batch_size=32, context_len=32, responce_len=32, persona_len=32):
        self.path = path
        self.batch_size = batch_size
        self.tokenizer_func = tokenizer_func
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.responce_len = responce_len
        self.persona_len = persona_len
    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as data:
            batch = None
            for line in data:
                x = json.loads(line)
                if batch is None:
                    batch = {k:[x[k]] for k in x}
                else:
                    for k in x:
                        batch[k].append(x[k])
                if len(batch['context']) == self.batch_size:
                    if self.tokenizer_func:
                        for k in batch:
                            if k == 'context':
                                max_len = self.context_len
                                join_token = self.tokenizer.end_of_masage_token
                            elif k == 'responce':
                                max_len = self.responce_len
                                join_token = False
                            elif k == 'persona':
                                max_len = self.persona_len
                                join_token = self.tokenizer.end_of_persona_sentence_token
                            else:
                                continue
                            batch[k] = self.tokenizer_func(batch[k], tokenizer=self.tokenizer, max_len=max_len, join_token=join_token)
                    yield batch, batch.pop('label')
                    batch = None

    def __next__(self):
        return json.loads(self.data.__next__())

    def __len__(self):
        c = 0
        for _ in self:
            c+=1
            print(c)
        return c

def clf(inp, tokenizer_func, tokenizer=False, context_len=32, responce_len=32, persona_len=32):
    batch = None
    for line in inp:
        x = json.loads(line)
        if batch is None:
            batch = {k:[x[k]] for k in x}
        else:
            for k in x:
                batch[k].append(x[k])
    if tokenizer_func:
        for k in batch:
            if k == 'context':
                max_len = context_len
                join_token = tokenizer.end_of_masage_token
            elif k == 'responce':
                max_len = responce_len
                join_token = False
            elif k == 'persona':
                max_len = persona_len
                join_token = tokenizer.end_of_persona_sentence_token
            else:
                continue
            batch[k] = tokenizer_func(batch[k], tokenizer=tokenizer, max_len=max_len, join_token=join_token)
    return batch, batch.pop('label')

class TolokaTorchDataset(torch.utils.data.Dataset):
    def __init__(self, path): #, tokenizer_func=False, tokenizer=False, batch_size=32, context_len=32, responce_len=32, persona_len=32
        with open(path, 'r', encoding='utf-8') as data:
            self.data = data.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        



#train1 = TolokaLazyDataset(data_load_config['proc_data_path'], tokenizer_func=tokenize_ru, tokenizer=bert_tokenizer, batch_size=64, context_len=32, responce_len=32, persona_len=32)

train = TolokaTorchDataset(data_load_config['proc_data_path'])
train = torch.utils.data.DataLoader(train, batch_size=32,
                        shuffle=True, num_workers=0, collate_fn=lambda x: clf(x, tokenizer_func=tokenize_ru, tokenizer=bert_tokenizer, context_len=32, responce_len=32, persona_len=32))

###

#@title Текст заголовка по умолчанию
def cprint(*args):
    text = ""
    for arg in args:
        text += "{0} ".format(arg)
    logging.info(text)
    
def match(model, matching_method, x, y, x_mask, y_mask):
    # Multi-hop Co-Attention
    # x: (batch_size, m, hidden_size)
    # y: (batch_size, n, hidden_size)
    # x_mask: (batch_size, m)
    # y_mask: (batch_size, n)
    assert x.dim() == 3 and y.dim() == 3
    assert x_mask.dim() == 2 and y_mask.dim() == 2
    assert x_mask.shape == x.shape[:2] and y_mask.shape == y.shape[:2]
    m = x.shape[1]
    n = y.shape[1]

    attn_mask = torch.bmm(x_mask.unsqueeze(-1), y_mask.unsqueeze(1)) # (batch_size, m, n)
    attn = torch.bmm(x, y.transpose(1,2)) # (batch_size, m, n)
    model.attn = attn
    model.attn_mask = attn_mask
    
    x_to_y = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=2) # (batch_size, m, n)
    y_to_x = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=1).transpose(1,2) # # (batch_size, n, m)
    
    # x_attended, y_attended = None, None # no hop-1
    x_attended = torch.bmm(x_to_y, y) # (batch_size, m, hidden_size)
    y_attended = torch.bmm(y_to_x, x) # (batch_size, n, hidden_size)

    # x_attended_2hop, y_attended_2hop = None, None # no hop-2
    y_attn = torch.bmm(y_to_x.mean(dim=1, keepdim=True), x_to_y) # (batch_size, 1, n) # true important attention over y
    x_attn = torch.bmm(x_to_y.mean(dim=1, keepdim=True), y_to_x) # (batch_size, 1, m) # true important attention over x

    # truly attended representation
    x_attended_2hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
    y_attended_2hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)

    # # hop-3
    # y_attn, x_attn = torch.bmm(x_attn, x_to_y), torch.bmm(y_attn, y_to_x) # (batch_size, 1, n) # true important attention over y
    # x_attended_3hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
    # y_attended_3hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)
    # x_attended_2hop = torch.cat([xbatch_context_embhopbatch_context_embd_3hop], dim=-1)
    # y_attended_2hop = torch.cat([y_attended_2hop, y_attended_3hop], dim=-1)

    x_attended = x_attended, x_attended_2hop
    y_attended = y_attended, y_attended_2hop

    return x_attended, y_attended


def aggregate(model, aggregation_method, x, x_mask):
    # x: (batch_size, seq_len, emb_size)
    # x_mask: (batch_size, seq_len)
    assert x.dim() == 3 and x_mask.dim() == 2
    assert x.shape[:2] == x_mask.shape
    # batch_size, seq_len, emb_size = x.shape

    if aggregation_method == "mean":
        return (x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (batch_size, emb_size)

    if aggregation_method == "max":
        return x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0] # (batch_size, emb_size)

    if aggregation_method == "mean_max":
        return torch.cat([(x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1), \
            x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0]], dim=-1) # (batch_size, 2*emb_size)


def fuse(model, matching_method, aggregation_method, batch_x_emb, batch_y_emb, batch_persona_emb, \
    batch_x_mask, batch_y_mask, batch_persona_mask, batch_size, num_candidates):
    
    batch_x_emb, batch_y_emb_context = match(model, matching_method, batch_x_emb, batch_y_emb, batch_x_mask, batch_y_mask)
    # batch_x_emb: ((batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size))
    # batch_y_emb_context: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)
    
    # hop 2 results
    batch_x_emb_2hop = batch_x_emb[1]
    batch_y_emb_context_2hop = batch_y_emb_context[1]
    
    # mean_max aggregation for the 1st hop result
    batch_x_emb = aggregate(model, aggregation_method, batch_x_emb[0], batch_x_mask) # batch_x_emb: (batch_size*num_candidates, 2*emb_size)
    batch_y_emb_context = aggregate(model, aggregation_method, batch_y_emb_context[0], batch_y_mask) # batch_y_emb_context: (batch_size*num_candidates, 2*emb_size)

    if batch_persona_emb is not None:
        batch_persona_emb, batch_y_emb_persona = match(model, matching_method, batch_persona_emb, batch_y_emb, batch_persona_mask, batch_y_mask)
        # batch_persona_emb: (batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size)
        # batch_y_emb_persona: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)

        batch_persona_emb_2hop = batch_persona_emb[1]
        batch_y_emb_persona_2hop = batch_y_emb_persona[1]

        # # no hop-1
        # return torch.bmm(torch.cat([batch_x_emb_2hop, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
        #             torch.cat([batch_y_emb_context_2hop, batch_y_emb_persona_2hop], dim=-1)\
        #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)

        
        batch_persona_emb = aggregate(model, aggregation_method, batch_persona_emb[0], batch_persona_mask) # batch_persona_emb: (batch_size*num_candidates, 2*emb_size)
        batch_y_emb_persona = aggregate(model, aggregation_method, batch_y_emb_persona[0], batch_y_mask) # batch_y_emb_persona: (batch_size*num_candidates, 2*emb_size)

        # # no hop-2
        # return torch.bmm(torch.cat([batch_x_emb, batch_persona_emb], dim=-1).unsqueeze(1), \
        #             torch.cat([batch_y_emb_context, batch_y_emb_persona], dim=-1)\
        #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)
        return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop, batch_persona_emb, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
                    torch.cat([batch_y_emb_context, batch_y_emb_context_2hop, batch_y_emb_persona, batch_y_emb_persona_2hop], dim=-1)\
                        .unsqueeze(-1)).reshape(batch_size, num_candidates)
    else:
        return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop], dim=-1).unsqueeze(1), \
                    torch.cat([batch_y_emb_context, batch_y_emb_context_2hop], dim=-1)\
                        .unsqueeze(-1)).reshape(batch_size, num_candidates)


def dot_product_loss(batch_x_emb, batch_y_emb):
    """
        if batch_x_emb.dim() == 2:
            # batch_x_emb: (batch_size, emb_size)
            # batch_y_emb: (batch_size, emb_size)
        
        if batch_x_emb.dim() == 3:
            # batch_x_emb: (batch_size, batch_size, emb_size), the 1st dim is along examples and the 2nd dim is along candidates
            # batch_y_emb: (batch_size, emb_size)
    """
    batch_size = batch_x_emb.size(0)
    targets = torch.arange(batch_size, device=batch_x_emb.device)

    if batch_x_emb.dim() == 2:
        dot_products = batch_x_emb.mm(batch_y_emb.t())
    elif batch_x_emb.dim() == 3:
        dot_products = torch.bmm(batch_x_emb, batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2))[:, targets, targets] # (batch_size, batch_size)
    
    # dot_products: [batch, batch]
    log_prob = torch.nn.functional.log_softmax(dot_products, dim=1)
    loss = torch.nn.functional.nll_loss(log_prob, targets)
    print(targets)
    print(log_prob.max(dim=1)[1])
    nb_ok = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, nb_ok

def train_epoch(data_iter, models, optimizers, schedulers, gradient_accumulation_steps, device, fp16, amp, \
    apply_interaction, matching_method, aggregation_method): #num_personas
    models = [i.to(device) for i in models]
    epoch_loss = []
    ok = 0
    total = 0
    print_every = 500
    context_model, response_model, persona_model = models[0], models[0], models[0]
    for optimizer in optimizers:
        optimizer.zero_grad()
    for i, batch in enumerate(data_iter):
        ######
        x, y = batch
        batch_context = x['context'].to(device)
        batch_responce = x['responce'].to(device)
        batch_persona = x['persona'].to(device)
        
        
        output_context = context_model(**batch_context)
        output_responce = response_model(**batch_responce)
        
        if apply_interaction:
            # batch_context_mask = batch[0].ne(0).float()
            # batch_responce_mask = batch[3].ne(0).float()
            batch_context_mask = batch_context['attention_mask'].float()
            batch_responce_mask = batch_responce['attention_mask'].float()
            batch_context_emb = output_context[0] # (batch_size, context_len, emb_size)
            batch_responce_emb = output_responce[0] # (batch_size, sent_len, emb_size)
            batch_size, sent_len, emb_size = batch_responce_emb.shape

            batch_persona_emb = None
            batch_persona_mask = None
            num_candidates = batch_size
            if True: #has_persona
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch_persona['attention_mask'].float()
                output_persona = persona_model(**batch_persona)
                batch_persona_emb = output_persona[0] # (batch_size, persona_len, emb_size)

                batch_persona_emb = batch_persona_emb.repeat_interleave(num_candidates, dim=0)
                batch_persona_mask = batch_persona_mask.repeat_interleave(num_candidates, dim=0)

            batch_context_emb = batch_context_emb.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len, emb_size)
            batch_context_mask = batch_context_mask.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
            
            # interaction
            # context-response attention
            batch_responce_emb = batch_responce_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, sent_len, emb_size) # (batch_size*num_candidates, sent_len, emb_size)
            batch_responce_mask = batch_responce_mask.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, sent_len) # (batch_size*num_candidates, sent_len)
            logits = fuse(context_model, matching_method, aggregation_method, \
                batch_context_emb, batch_responce_emb, batch_persona_emb, batch_context_mask, batch_responce_mask, batch_persona_mask, batch_size, num_candidates)
            
            # compute loss
            targets = torch.arange(batch_size, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
        else:
            batch_context_emb = output_context[0].mean(dim=1) # batch_context_emb: (batch_size, emb_size)
            batch_responce_emb = output_responce[0].mean(dim=1)

            if True: #has_persona
                output_persona = persona_model(**batch_persona)
                batch_persona_emb = output_persona[0].mean(dim=1)
                batch_context_emb = (batch_context_emb + batch_persona_emb)/2
            
            # compute loss
            loss, num_ok = dot_product_loss(batch_context_emb, batch_responce_emb)
        
        ok += num_ok.item()
        total += batch_context['attention_mask'].shape[0]

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if (i+1) % gradient_accumulation_steps == 0:
            for model, optimizer, scheduler in zip(models, optimizers, schedulers):
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                
                # clear grads here
                for optimizer in optimizers:
                    optimizer.zero_grad()
        epoch_loss.append(loss.item())

        if i%print_every == 0:
            # cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            # cprint("accuracy: ", ok/total)
            print("loss: ", np.mean(epoch_loss[-print_every:]))
            print("accuracy: ", ok/total)

    acc = ok/total
    return np.mean(epoch_loss), (acc, 0, 0)
  
  ###
  
  for epoch in range(epochs):
        print("Epoch", epoch+1)
        # training
        for model in models:
            model.train()
            
        train = tqdm(train, desc="Iteration")
        train_loss, (train_acc, _, _) = train_epoch(data_iter=train, 
                                                    models=models, optimizers=optimizers, 
                                                    schedulers=schedulers, 
                                                    gradient_accumulation_steps=gradient_accumulation_steps, 
                                                    device=device, fp16=fp16, 
                                                    amp=amp, apply_interaction=apply_interaction, 
                                                    matching_method=matching_method, 
                                                    aggregation_method=aggregation_method)
        epoch_train_losses.append(train_loss)

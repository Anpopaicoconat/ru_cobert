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

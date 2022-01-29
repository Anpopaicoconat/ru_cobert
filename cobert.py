import json
import csv
import math
import torch
import transformers
import numpy as np

from util import count_parameters, compute_metrics, compute_metrics_from_logits

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
    if aggregation_method == "cls":
        return x[:,0] # (batch_size, emb_size)
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
    nb_ok = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, nb_ok

def train_epoch(data_iter, models, optimizers, schedulers, gradient_accumulation_steps, device, fp16, amp, \
    apply_interaction, matching_method, aggregation_method): #num_personas
    models = [i.to(device) for i in models]
    epoch_loss = []
    ok = 0
    total = 0
    print_every = 100
    context_model, response_model, persona_model = models[0], models[0], models[0]
    for optimizer in optimizers:
        optimizer.zero_grad()
    for i, batch in enumerate(data_iter):
        ######
        x, y = batch
        # batch_context = x['context'].to(device)
        # batch_responce = x['responce'].to(device)
        # batch_persona = x['persona'].to(device)
        batch_context = {k:x['context'][k].to(device) for k in x['context']}
        batch_responce = {k:x['responce'][k].to(device) for k in x['responce']}
        batch_persona = {k:x['persona'][k].to(device) for k in x['persona']}
        
        
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
            # print('targets', targets.long())
            # print('logits', logits.float().argmax(dim=1))
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

        if  i and i%print_every == 0:
            # cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            # cprint("accuracy: ", ok/total)
            print("loss: ", np.mean(epoch_loss[-print_every:]))
            print("accuracy: ", ok/total)

    acc = ok/total
    return np.mean(epoch_loss), (acc, 0, 0)

def evaluate_epoch(data_iter, models, gradient_accumulation_steps, device, epoch, \
    apply_interaction, matching_method, aggregation_method):
    epoch_loss = []
    ok = 0
    total = 0
    recall = []
    MRR = []
    print_every = 100
    context_model, response_model, persona_model = models[0], models[0], models[0]
    
    for batch_idx, batch in enumerate(data_iter):
        x, y = batch
        batch_context = x['context'].to(device)
        batch_responce = x['responce'].to(device)
        batch_persona = x['persona'].to(device)
        
        # get context embeddings in chunks due to memory constraint
        batch_size = batch_context['input_ids'].shape[0]
        chunk_size = 20
        num_chunks = math.ceil(batch_size/chunk_size)

        if apply_interaction:
            batch_context_mask = batch_context['attention_mask'].float()
            batch_responce_mask = batch_responce['attention_mask'].float()
            
            batch_context_emb = []
            batch_context_pooled_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_context = {
                        "input_ids": batch_context['input_ids'][i*chunk_size: (i+1)*chunk_size], 
                        "attention_mask": batch_context['attention_mask'][i*chunk_size: (i+1)*chunk_size], 
                        "token_type_ids": batch_context['token_type_ids'][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_context = context_model(**mini_batch_context)
                    batch_context_emb.append(mini_output_context[0]) # [(chunk_size, seq_len, emb_size), ...]
                    batch_context_pooled_emb.append(mini_output_context[1])
                batch_context_emb = torch.cat(batch_context_emb, dim=0) # (batch_size, seq_len, emb_size)
                batch_context_pooled_emb = torch.cat(batch_context_pooled_emb, dim=0)
                emb_size = batch_context_emb.shape[-1]

            batch_persona_mask = batch_persona['attention_mask'].float()
            batch_persona_emb = []
            batch_persona_pooled_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_persona = {
                        "input_ids": batch_persona['input_ids'][i*chunk_size: (i+1)*chunk_size], 
                        "attention_mask": batch_persona['attention_mask'][i*chunk_size: (i+1)*chunk_size], 
                        "token_type_ids": batch_persona['token_type_ids'][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_persona = persona_model(**mini_batch_persona)

                    # [(chunk_size, emb_size), ...]
                    batch_persona_emb.append(mini_output_persona[0])
                    batch_persona_pooled_emb.append(mini_output_persona[1])

                batch_persona_emb = torch.cat(batch_persona_emb, dim=0)
                batch_persona_pooled_emb = torch.cat(batch_persona_pooled_emb, dim=0)

        with torch.no_grad():
            output_responce = response_model(**batch_responce)
            batch_responce_emb = output_responce[0]
        batch_size, sent_len, emb_size = batch_responce_emb.shape

        # interaction
        # context-response attention
        num_candidates = batch_size
        
        with torch.no_grad():
            # evaluate per example
            logits = []
            for i in range(batch_size):
                context_emb = batch_context_emb[i:i+1].repeat_interleave(num_candidates, dim=0) # (num_candidates, context_len, emb_size)
                context_mask = batch_context_mask[i:i+1].repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
                persona_emb, persona_mask = None, None
                persona_emb = batch_persona_emb[i:i+1].repeat_interleave(num_candidates, dim=0)
                persona_mask = batch_persona_mask[i:i+1].repeat_interleave(num_candidates, dim=0)

                logits_single = fuse(context_model, matching_method, aggregation_method, \
                    context_emb, batch_responce_emb, persona_emb, context_mask, batch_responce_mask, persona_mask, 1, num_candidates).reshape(-1)
                
                logits.append(logits_single)
            logits = torch.stack(logits, dim=0)
            
            # compute loss
            targets = torch.arange(batch_size, dtype=torch.long, device=batch_context['input_ids'].device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        # print('targets', targets.long())
        # print('logits', logits.float().argmax(dim=1))
        num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
        valid_recall, valid_MRR = compute_metrics_from_logits(logits, targets)
        
        ok += num_ok.item()
        total += batch_context['input_ids'].shape[0]

        # compute valid recall
        recall.append(valid_recall)
        MRR.append(valid_MRR)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        epoch_loss.append(loss.item())

        if batch_idx and batch_idx%print_every == 0:
            print("loss: ", np.mean(epoch_loss[-print_every:]))
            print("valid recall: ", np.mean(recall[-print_every:], axis=0))
            print("valid MRR: ", np.mean(MRR[-print_every:], axis=0))

    acc = ok/total
    # compute recall for validation dataset
    recall = np.mean(recall, axis=0)
    MRR = np.mean(MRR)
    return np.mean(epoch_loss), (acc, recall, MRR)

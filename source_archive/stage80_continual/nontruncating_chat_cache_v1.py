"""Explicit answer-only encoding; overlength data fail instead of truncating.

This is a new prospective helper. Existing pinned 64-token experiments remain
unchanged. The chosen future runtime tokenizer must be checked again at admission.
"""
import torch
from model_support import format_chat


def encode_items(tokenizer, items, max_length, context_capacity):
    if not isinstance(max_length,int) or not 1 < max_length <= context_capacity:
        raise ValueError('Declare a valid maximum within the actual model context')
    if not items:
        raise ValueError('Nonempty training rows required')
    encoded=[]
    for item in items:
        question=item['train_q']; answer=item['answer']
        if not isinstance(question,str) or not question.strip() or not isinstance(answer,str) or not answer.strip():
            raise ValueError('A nonempty teacher question and answer are required')
        prompt=tokenizer(format_chat(tokenizer,question),add_special_tokens=False)['input_ids']
        full=tokenizer(format_chat(tokenizer,question,answer),add_special_tokens=False)['input_ids']
        if len(full)>max_length:
            raise ValueError(f'Teacher row {item.get("id")} has {len(full)} tokens, declared maximum {max_length}; no truncation allowed')
        if not prompt or full[:len(prompt)]!=prompt or len(full)<=len(prompt):
            raise ValueError('Prompt prefix changed or supervised answer tokens are missing')
        labels=[-100]*len(prompt)+full[len(prompt):]
        encoded.append((torch.tensor(full,dtype=torch.long),torch.tensor(labels,dtype=torch.long)))
    return encoded


def collate(encoded,indices,pad_id,device):
    if not indices:
        raise ValueError('Nonempty minibatch required')
    selected=[encoded[i] for i in indices]
    if any(ids.ndim!=1 or labels.shape!=ids.shape or ids.dtype!=torch.long or labels.dtype!=torch.long
           for ids,labels in selected):
        raise ValueError('Cache tensor dimensions/dtypes are invalid')
    length=max(len(ids) for ids,_ in selected)
    ids=torch.full((len(selected),length),pad_id,dtype=torch.long)
    labels=torch.full_like(ids,-100)
    for i,(row,target) in enumerate(selected):
        ids[i,:len(row)]=row; labels[i,:len(target)]=target
    return ids.to(device),labels.to(device)

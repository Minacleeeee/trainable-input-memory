"""Explicit2D masked log-softmax NLL, avoiding large-vocabulary CE reductions."""
import torch
from torch.nn import functional as F


def supervised_nll(logits,labels):
    targets=labels[...,1:].contiguous().reshape(-1)
    valid=targets!=-100
    count=int(valid.detach().cpu().sum())
    if count<1: raise ValueError('Positive supervised count required')
    scores=logits[...,:-1,:].float().contiguous().reshape(-1,logits.shape[-1])
    values=-F.log_softmax(scores,dim=-1).gather(1,targets.clamp(min=0)[:,None]).squeeze(1)
    return (values*valid.to(values.dtype)).sum()/count

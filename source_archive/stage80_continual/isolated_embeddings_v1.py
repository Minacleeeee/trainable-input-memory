"""Prototype neural input parameter isolation, not an answer table or retrieval.

Freeze the existing network and vocabulary rows. Each preallocated new lexical
block is a neural parameter matrix; train one block at a time and freeze it
after completion. Answers still require the full frozen network forward pass.
Tokenizer configuration stores names/IDs only, never colors or training replies.
CPU-tested prototype only; DirectML compatibility and learning remain untested.
"""
import torch
from torch import nn
from torch.nn import functional as F


class IsolatedInputEmbedding(nn.Module):
    def __init__(self, original, initial_blocks):
        super().__init__()
        if not isinstance(original,nn.Embedding) or not initial_blocks:
            raise ValueError('Original neural embedding and nonempty fixed blocks required')
        if any(block.ndim!=2 or block.shape[1]!=original.embedding_dim or block.shape[0]<1
               for block in initial_blocks):
            raise ValueError('Invalid neural parameter block dimensions')
        self.original=original
        self.original.weight.requires_grad_(False)
        self.original_vocab_size=original.num_embeddings
        self.embedding_dim=original.embedding_dim
        self.blocks=nn.ParameterList([nn.Parameter(block.detach().clone().to(
            device=original.weight.device,dtype=torch.float32),requires_grad=False) for block in initial_blocks])
        self.num_embeddings=self.original_vocab_size+sum(p.shape[0] for p in self.blocks)
        self.active_block=None

    @property
    def weight(self):
        # Device/dtype inspection only. The original output head remains tied
        # to these frozen original rows; do not resize it to new input IDs.
        return self.original.weight

    def activate_block(self,index):
        if index is not None and (not isinstance(index,int) or not 0<=index<len(self.blocks)):
            raise ValueError('Invalid trainable block')
        for i,p in enumerate(self.blocks): p.requires_grad_(i==index)
        self.active_block=index

    def forward(self,input_ids):
        # Ordinary neural embedding selection. No question parser, label lookup,
        # answer routing or output correction exists in this module.
        old_ids=input_ids.clamp(max=self.original_vocab_size-1)
        old_vectors=self.original(old_ids)
        new_ids=(input_ids-self.original_vocab_size).clamp(min=0)
        new_vectors=F.embedding(new_ids,torch.cat(list(self.blocks),dim=0)).to(old_vectors.dtype)
        return torch.where((input_ids>=self.original_vocab_size).unsqueeze(-1),new_vectors,old_vectors)


def freeze_network_except_input_block(model,embedding,index):
    for p in model.parameters(): p.requires_grad_(False)
    embedding.activate_block(index)
    trainable=[p for p in model.parameters() if p.requires_grad]
    if len(trainable)!=1 or trainable[0] is not embedding.blocks[index]:
        raise ValueError('Only the selected neural input matrix may learn')
    return trainable


def add_input_names(tokenizer,names,original_vocab_size):
    """Install predeclared lexical IDs; input inventory contains no answers."""
    from tokenizers import AddedToken
    if (len(tokenizer)!=original_vocab_size or not names or len(names)!=len(set(names))
            or any(not isinstance(n,str) or not n or any(c.isspace() for c in n) for n in names)):
        raise ValueError('Fixed new names and original tokenizer size required')
    if any(n in tokenizer.get_vocab() for n in names):
        raise ValueError('New lexical names must not reuse original vocabulary IDs')
    added=tokenizer.add_tokens([AddedToken(n,single_word=True,normalized=False) for n in names])
    if added!=len(names) or len(tokenizer)!=original_vocab_size+len(names):
        raise ValueError('New lexical inventory changed shape')
    for i,name in enumerate(names):
        if tokenizer(name,add_special_tokens=False)['input_ids']!=[original_vocab_size+i]:
            raise ValueError('New entity is not exactly its predeclared input ID')
    return {'original_vocab_size':original_vocab_size,'new_input_rows':len(names),
            'names':names,'contains_answer_labels':False,'output_vocabulary_resized':False}

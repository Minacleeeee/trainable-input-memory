"""Same neural input wrapper, efficient lexical installation for future runs.

Version1 probe/auditor stay pinned. This caches the old vocabulary once and
checks predeclared names in a batch; it does not alter tokens or learned behavior.
"""
from isolated_embeddings_v1 import IsolatedInputEmbedding,freeze_network_except_input_block


def add_input_names(tokenizer,names,original_vocab_size):
    from tokenizers import AddedToken
    if (len(tokenizer)!=original_vocab_size or not names or len(names)!=len(set(names))
            or any(not isinstance(n,str) or not n or any(c.isspace() for c in n) for n in names)):
        raise ValueError('Fixed new names and original tokenizer size required')
    vocabulary=tokenizer.get_vocab()
    if any(n in vocabulary for n in names):
        raise ValueError('New lexical names must not reuse original vocabulary IDs')
    added=tokenizer.add_tokens([AddedToken(n,single_word=True,normalized=False) for n in names])
    if added!=len(names) or len(tokenizer)!=original_vocab_size+len(names):
        raise ValueError('New lexical inventory changed shape')
    encoded=tokenizer(names,add_special_tokens=False)['input_ids']
    if encoded!=[[original_vocab_size+i] for i in range(len(names))]:
        raise ValueError('New entities do not match predeclared lexical IDs')
    return {'original_vocab_size':original_vocab_size,'new_input_rows':len(names),
            'names':names,'contains_answer_labels':False,'output_vocabulary_resized':False}

"""Append label-blind neural input capacity without changing existing vectors."""
import torch


def extend_saved_blocks(saved, lexical, names, appended_initial):
    if set(saved) != {'blocks'} or len(saved['blocks']) != 10:
        raise ValueError('Exact ten-bank source checkpoint required')
    if (lexical.get('original_vocab_size') != 49152 or lexical.get('new_input_rows') != 5120
            or lexical.get('contains_answer_labels') is not False
            or lexical.get('output_vocabulary_resized') is not False
            or len(lexical.get('names', [])) != 5120):
        raise ValueError('Exact label-free original5120 lexical inventory required')
    if (len(names) != 10240 or names[:5120] != lexical['names'] or len(set(names)) != 10240
            or any(not isinstance(name, str) or not name or any(c.isspace() for c in name) for name in names)):
        raise ValueError('All original lexical IDs must remain an unchanged prefix')
    old = saved['blocks']
    if any(tuple(p.shape) != (512, 2048) or p.dtype != torch.float32
           or p.device.type != 'cpu' or not torch.isfinite(p).all() for p in old):
        raise ValueError('Finite source512x2048 fp32 CPU bank matrices required')
    if (tuple(appended_initial.shape) != (5120, 2048) or appended_initial.dtype != torch.float32
            or appended_initial.device.type != 'cpu' or not torch.isfinite(appended_initial).all()):
        raise ValueError('Finite label-blind appended5120x2048 CPU means required')
    blocks = [p.detach().clone() for p in old] + [p.clone() for p in appended_initial.detach().split(512)]
    assert len(blocks) == 20 and all(torch.equal(blocks[i], old[i]) for i in range(10))
    return {'blocks': blocks}

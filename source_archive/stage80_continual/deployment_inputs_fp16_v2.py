"""Frozen fp16 deployment input parameters; never a training/resume format.

Neural token embedding selection only. Runtime restoration reads lexical IDs,
configuration and weights, never teaching archives or evaluation answers.
Version 2 preserves the qualified DirectML fp32-to-fp16 cast, rather than CPU round-to-nearest. Full-generation equivalence still requires an actual audit.
"""
from pathlib import Path
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F


class DeploymentInputEmbeddingFP16(nn.Module):
    def __init__(self, original, blocks):
        super().__init__()
        if not isinstance(original, nn.Embedding) or original.weight.dtype != torch.float16:
            raise ValueError('A frozen fp16 original embedding is required')
        if not blocks or any(b.ndim != 2 or b.shape[0] < 1 or
                             b.shape[1] != original.embedding_dim for b in blocks):
            raise ValueError('Nonempty dimension-matched neural banks are required')
        # Validate CPU values once at restore, not by inspecting question content.
        converted = []
        for block in blocks:
            raw = block.detach().cpu()
            if not bool(torch.isfinite(raw).all()):
                raise ValueError('Nonfinite source parameters')
            # Reproduce the qualified GPU forward conversion before storing
            # frozen half tensors. No original fp32 bank remains resident.
            if raw.dtype == torch.float32:
                raw = raw.to(original.weight.device).to(torch.float16).cpu()
            elif raw.dtype != torch.float16:
                raise ValueError('Only fp32 masters or frozen fp16 deployment banks')
            converted.append(raw)
        if not all(bool(torch.isfinite(b).all()) for b in converted):
            raise ValueError('Nonfinite or overflowing deployment parameters')
        self.original = original
        self.original.weight.requires_grad_(False)
        self.original_vocab_size = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.blocks = nn.ParameterList([
            nn.Parameter(b.clone().to(original.weight.device), requires_grad=False)
            for b in converted])
        self.num_embeddings = self.original_vocab_size + sum(b.shape[0] for b in self.blocks)
        self.active_block = None

    @property
    def weight(self):
        return self.original.weight

    def activate_block(self, index):
        if index is not None:
            raise RuntimeError('Deployment precision is not qualified for training or resume')
        for block in self.blocks:
            block.requires_grad_(False)

    def forward(self, input_ids):
        old_ids = input_ids.clamp(max=self.original_vocab_size - 1)
        old_vectors = self.original(old_ids)
        new_ids = (input_ids - self.original_vocab_size).clamp(min=0)
        new_vectors = F.embedding(new_ids, torch.cat(list(self.blocks), dim=0))
        return torch.where((input_ids >= self.original_vocab_size).unsqueeze(-1),
                           new_vectors, old_vectors)


def restore(device, checkpoint):
    """Restore deployment tensors directly; no resident fp32 learned banks.

    The caller must independently qualify the scientific source and eventual
    generation parity. This loader intentionally has no path to those answers.
    It accepts the original immutable fp32 weight file for the prospective probe;
    this alone does not reduce that file's disk size or delete training masters.
    """
    from engine import pin_backbone, load
    from model_support import load_tokenizer
    from isolated_inputs_v2 import add_input_names
    cp = Path(checkpoint)
    config = json.loads((cp/'inference_config.json').read_text(encoding='utf-8'))
    if pin_backbone() != config['backbone']:
        raise ValueError('Frozen backbone fingerprint changed')
    for name, expected in config['frozen_adapter_hashes'].items():
        actual = hashlib.sha256((Path(config['frozen_adapter_path'])/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen maintenance checkpoint changed')
    tokenizer = load_tokenizer()
    model = load('B', device, config['frozen_adapter_path'], seed=42)
    lexical = json.loads((cp/'lexical_inventory.json').read_text(encoding='utf-8'))
    if add_input_names(tokenizer, lexical['names'], lexical['original_vocab_size']) != lexical:
        raise ValueError('Lexical IDs changed')
    saved = torch.load(cp/'input_parameters.pt', map_location='cpu', weights_only=True)
    if set(saved) != {'blocks'}:
        raise ValueError('Unexpected runtime parameter payload')
    embedding = DeploymentInputEmbeddingFP16(model.get_input_embeddings(), saved['blocks'])
    if sum(b.shape[0] for b in embedding.blocks) != lexical['new_input_rows']:
        raise ValueError('Lexical inventory and parameter rows disagree')
    model.set_input_embeddings(embedding)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, tokenizer

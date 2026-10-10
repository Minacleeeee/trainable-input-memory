"""Always-active low-rank neural projection updates; no task routing or labels."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

CONFIG = {'layer_index': 23, 'targets': ['q_proj', 'v_proj'], 'rank': 2,
          'alpha': 4.0, 'initialization_seed': 20261005, 'A_std': .01,
          'dtype': 'float32', 'always_active': True, 'dropout': 0.0}


class SharedProjection(nn.Module):
    def __init__(self, original, rank, alpha, seed, std):
        super().__init__()
        self.original = original
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank
        gen = torch.Generator(device='cpu').manual_seed(seed)
        initial = torch.randn(rank, self.in_features, generator=gen, dtype=torch.float32) * std
        device = original.weight.device
        self.skill_down = nn.Parameter(initial.to(device))
        self.skill_up = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=device))
        for p in self.original.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        base = self.original(x)
        residual = F.linear(F.linear(x.float(), self.skill_down), self.skill_up) * self.scaling
        return base + residual.to(base.dtype)


def attach(model, config=CONFIG, saved=None):
    if config != CONFIG:
        raise ValueError('Pinned shared module configuration required')
    suffix = 'layers.' + str(config['layer_index']) + '.self_attn'
    matches = [m for n, m in model.named_modules() if n.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('Exactly one declared attention layer required')
    attention = matches[0]; modules = {}
    for i, name in enumerate(config['targets']):
        original = getattr(attention, name)
        if isinstance(original, SharedProjection):
            raise ValueError('Never wrap twice')
        if list(getattr(original, 'active_adapters', [])) != ['single']:
            raise ValueError('Frozen original maintenance adapter must remain active')
        if original.in_features != 2048 or original.out_features != 2048:
            raise ValueError('Unexpected projection shape')
        module = SharedProjection(original, config['rank'], config['alpha'],
                                  config['initialization_seed'] + i, config['A_std'])
        if saved is not None:
            state = saved[name]
            assert set(state) == {'A', 'B'}
            for p, key in [(module.skill_down, 'A'), (module.skill_up, 'B')]:
                t = state[key]
                if t.shape != p.shape or t.dtype != torch.float32 or not torch.isfinite(t).all():
                    raise ValueError('Corrupt shared neural parameters')
                with torch.no_grad(): p.copy_(t.to(p.device))
        setattr(attention, name, module); modules[name] = module
    if saved is not None and set(saved) != set(modules):
        raise ValueError('Unexpected shared module keys')
    return modules


def export(modules):
    return {n: {'A': m.skill_down.detach().cpu().clone(), 'B': m.skill_up.detach().cpu().clone()}
            for n, m in modules.items()}


def configure_trainable(model, embedding, modules, arm):
    if arm not in ('input', 'shared'):
        raise ValueError('Declared arm required')
    for p in model.parameters(): p.requires_grad_(False)
    embedding.activate_block(20)
    params = {'name': [embedding.blocks[20]], 'shared': []}
    if arm == 'shared':
        for m in modules.values():
            for p in (m.skill_down, m.skill_up):
                p.requires_grad_(True); params['shared'].append(p)
    elif modules:
        raise ValueError('Input baseline must not contain shared modules')
    expected = {id(p) for ps in params.values() for p in ps}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected or sum(p.numel() for ps in params.values() for p in ps) != (18432 if arm == 'shared' else 2048):
        raise ValueError('Incorrect trainable whitelist')
    return params


def maintenance_hashes(model):
    out = {}
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            if parameter.requires_grad:
                raise ValueError('Frozen maintenance adapter became trainable')
            normalized = name.replace('.original.', '.')
            if normalized in out: raise ValueError('Ambiguous frozen adapter name')
            out[normalized] = hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if not out: raise ValueError('Maintenance adapter missing')
    for module in model.modules():
        if hasattr(module, 'lora_A') and list(module.active_adapters) != ['single']:
            raise ValueError('Maintenance adapter activation changed')
    return out


def restore(device, checkpoint):
    """Reads only runtime configs, lexical IDs and learned weights."""
    import json
    from pathlib import Path
    from run_isolated_memory_pilot_v2 import restore as restore_inputs
    cp = Path(checkpoint)
    model, tok = restore_inputs(device, cp)
    config = json.loads((cp / 'shared_skill_config.json').read_text(encoding='utf-8'))
    if set(config) != {'arm', 'module_config'}: raise ValueError('Invalid runtime-only config')
    modules = {}
    if config['arm'] == 'shared':
        if config['module_config'] != CONFIG: raise ValueError('Changed module config')
        saved = torch.load(cp / 'shared_skill_parameters.pt', map_location='cpu', weights_only=True)
        modules = attach(model, CONFIG, saved)
    elif config != {'arm': 'input', 'module_config': None}:
        raise ValueError('Invalid baseline runtime config')
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return model, tok, modules

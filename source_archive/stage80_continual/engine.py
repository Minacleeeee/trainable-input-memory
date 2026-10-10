"""Trainable parametric memory. Inference receives only the current query."""
import gc
import math
import random
import torch
from model_support import (
    load_base, load_tokenizer, attach_single_adapter, attach_dual_adapters,
    load_saved_adapters, fast_config, generate_response, make_chat_batch,
)
from storage import exact_score


def pin_backbone():
    import model_support
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_support.MODEL_NAME, local_files_only=True)
    revision = getattr(cfg, '_commit_hash', None)
    if not revision:
        raise RuntimeError('Cached backbone revision is unavailable')
    model_support.MODEL_REVISION = revision
    return {'model':model_support.MODEL_NAME, 'revision':revision,
            'context_capacity':cfg.max_position_embeddings}


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def activate(model, names, train_target=None):
    model.base_model.set_adapter(list(names), inference_mode=train_target is None)
    for name, p in model.named_parameters():
        selected = ('lora_' in name and train_target is not None
                    and f'.{train_target}.' in name)
        p.requires_grad_(selected)
    model.train(train_target is not None)
    # Verify activation at every participating LoRA layer.
    for module in model.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'active_adapters'):
            if list(module.active_adapters) != list(names):
                raise RuntimeError('Inconsistent LoRA activation')


def save_restoring_device(model, path):
    device = next(model.parameters()).device
    mode = model.training
    try:
        model.cpu()
        for name, p in model.named_parameters():
            if 'lora_' in name and not torch.isfinite(p).all().item():
                raise FloatingPointError('Non-finite adapter: ' + name)
        model.save_pretrained(str(path))
    finally:
        model.to(device)
        model.train(mode)
    if model.get_input_embeddings().weight.device != device:
        raise RuntimeError('Save changed the model device')


def load(arch, device, checkpoint=None, seed=42):
    seed_all(seed)
    base = load_base(device)
    if checkpoint is None:
        model = attach_dual_adapters(base) if arch == 'C' else attach_single_adapter(base, arch)
    else:
        model = load_saved_adapters(base, str(checkpoint))
    model.to(device)
    names = ['fast', 'slow'] if arch == 'C' else ['single']
    activate(model, names)
    return model


def train(model, tok, qa, epochs, lr, device, seed, tag, warmup=0.0):
    if not qa:
        return {'steps': 0, 'examples': 0, 'supervised_tokens': 0, 'losses': []}
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('Training requested without trainable parameters')
    if model.get_input_embeddings().weight.device != device:
        raise RuntimeError('Training model/device mismatch before forward')
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0, foreach=False)
    rng = random.Random(seed)
    warm_steps = max(0, int(math.ceil(len(qa)/2)*epochs*warmup))
    steps = tokens = 0
    losses = []
    model.train()
    for epoch in range(epochs):
        order = list(range(len(qa)))
        rng.shuffle(order)
        total = count = 0
        for i in range(0, len(order), 2):
            batch = [qa[j] for j in order[i:i+2]]
            ids, labels = make_chat_batch(tok, batch, device)
            if ids.shape != labels.shape or not (labels != -100).any().item():
                raise RuntimeError('Answer masking/truncation produced invalid labels')
            for group in opt.param_groups:
                group['lr'] = lr * min(1.0, (steps+1)/max(1, warm_steps))
            loss = model(input_ids=ids, labels=labels).loss
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f'Non-finite loss in {tag}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not math.isfinite(float(norm.detach().cpu())):
                raise FloatingPointError(f'Non-finite gradients in {tag}')
            opt.step()
            opt.zero_grad(set_to_none=True)
            tokens += int((labels[:, 1:] != -100).sum().item())
            steps += 1
            total += value
            count += 1
        losses.append(total/count)
        print(f'{tag} ep={epoch+1}/{epochs} loss={losses[-1]:.5f}', flush=True)
    return {'steps': steps, 'examples': len(qa)*epochs,
            'supervised_tokens': tokens, 'losses': losses}


def evaluate(model, tok, items, question_key='eval_q'):
    model.eval()
    rows = []
    for i, item in enumerate(items):
        response = generate_response(model, tok, item[question_key], max_new_tokens=20)
        rows.append({'id': item['id'], 'type': item['type'],
                     'question': item[question_key], 'expected': item['answer'],
                     'response': response, 'correct': exact_score(response, item['answer'])})
        if (i+1) % 20 == 0:
            gc.collect()
    return rows


def accuracy(rows):
    return sum(r['correct'] for r in rows)/len(rows) if rows else None


def reset_fast(model, device, seed):
    model.delete_adapter('fast')
    seed_all(seed)
    model.add_adapter('fast', fast_config())
    model.to(device)
    activate(model, ['fast', 'slow'])

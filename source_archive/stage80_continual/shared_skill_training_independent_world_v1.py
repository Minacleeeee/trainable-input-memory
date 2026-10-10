# Fresh version derived from shared_skill_training_v1.py; original production files remain immutable.
"""Versioned two-group name/shared-module trainer with actual exposure budgets."""
from collections import Counter
import math
import random
import time
from storage import atomic_json
from deadline import check_deadline, DeadlineReached


def train(model, tok, cache, training, parameter_groups, device, profile, run_path, stop_at):
    params = [p for ps in parameter_groups.values() for p in ps]
    import torch
    from cached_batches import collate
    from adamw_no_lerp import AdamWNoLerp
    from supervised_nll_v3 import supervised_nll
    phases = {}
    for phase in ('teach', 'sleep'):
        seed = profile['training_seeds'][phase]
        epochs = profile[phase + '_epochs']
        steps_total = math.ceil(len(training) / profile['batch']) * epochs
        specs = [{'params': ps, 'lr': profile[phase + '_' + name + '_lr'], 'role': name}
                 for name, ps in parameter_groups.items() if ps]
        optimizer = AdamWNoLerp(specs, lr=profile[phase + '_name_lr'], weight_decay=0)
        rng = random.Random(seed)
        steps = tokens = 0
        presentations = Counter()
        losses = []
        first_check = None
        started = time.perf_counter()
        try:
            for epoch in range(epochs):
                order = list(range(len(training)))
                rng.shuffle(order)
                epoch_loss = batches = 0
                for offset in range(0, len(order), profile['batch']):
                    check_deadline(stop_at, {'phase': phase, 'completed_updates': steps,
                                            'presentations': sum(presentations.values()), 'supervised_tokens': tokens,
                                            'training_sec': time.perf_counter() - started})
                    indices = order[offset:offset + profile['batch']]
                    ids, labels = collate(cache, indices, tok.pad_token_id, device)
                    warm = min(1., (steps + 1) / max(1, int(steps_total * .05)))
                    progress = steps / max(1, steps_total - 1)
                    factor = warm * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
                    for group in optimizer.param_groups:
                        group['lr'] = profile[phase + '_' + group['role'] + '_lr'] * factor
                    optimizer.zero_grad(set_to_none=True)
                    output = model(input_ids=ids)
                    loss = supervised_nll(output.logits, labels)
                    value = float(loss.detach().cpu())
                    if not math.isfinite(value): raise FloatingPointError('Nonfinite corrected NLL')
                    count = int((labels[:, 1:] != -100).sum().detach().cpu())
                    if not steps:
                        scores = output.logits.detach().cpu().float()[:, :-1, :]
                        targets = labels.detach().cpu()[:, 1:]
                        valid = targets != -100
                        cpu = float(-scores[valid].log_softmax(-1).gather(1, targets[valid].unsqueeze(1)).mean())
                        if abs(value - cpu) > 1e-4: raise ValueError('Actual minibatch NLL disagrees with independent CPU')
                        first_check = {'gpu_corrected_nll': value, 'independent_cpu_nll': cpu, 'supervised_tokens': count}
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(params, 1).detach().cpu())
                    if not math.isfinite(norm): raise FloatingPointError('Nonfinite skill/name gradient')
                    optimizer.step()
                    steps += 1
                    tokens += count
                    presentations.update(training[i]['id'] for i in indices)
                    epoch_loss += value
                    batches += 1
                    if steps % 100 == 0:
                        atomic_json(run_path / 'in_progress_updates.json', {'phase': phase, 'completed_phase_updates': steps, 'supervised_tokens': tokens})
                        print('SHARED_SKILL', phase, steps, '/', steps_total, 'nll', value, flush=True)
                losses.append(epoch_loss / batches)
                print('SHARED_SKILL', phase, 'epoch', epoch + 1, '/', epochs, 'nll', losses[-1], flush=True)
        except BaseException:
            atomic_json(run_path / 'interrupted_actual_budget.json',
                        {'completed_phases': phases, 'phase': phase, 'steps': steps,
                         'presentations': sum(presentations.values()), 'presentations_by_id': dict(presentations),
                         'supervised_tokens': tokens, 'training_sec': time.perf_counter() - started,
                         'completed_epoch_losses': losses, 'first_batch_independent_cpu_loss_check': first_check,
                         'partial_weights_not_qualified_or_reusable': True})
            raise
        phases[phase] = {'steps': steps, 'presentations': sum(presentations.values()), 'supervised_tokens': tokens,
                         'presentations_by_id': dict(presentations), 'epochs': epochs, 'seed': seed, 'losses': losses,
                         'training_sec': time.perf_counter() - started, 'first_batch_independent_cpu_loss_check': first_check,
                         'active_trainable_parameters': sum(p.numel() for p in params),
                         'optimizer': 'AdamWNoLerp',
                         'peak_lr_by_group': {n: profile[phase + '_' + n + '_lr'] for n, ps in parameter_groups.items() if ps},
                         'parameters_by_group': {n: sum(p.numel() for p in ps) for n, ps in parameter_groups.items() if ps}}
        atomic_json(run_path / 'in_progress_phase_budget.json', {'phases': phases})
    return phases

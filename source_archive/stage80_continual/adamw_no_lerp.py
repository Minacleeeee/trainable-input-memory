"""Experimental AdamW variant avoiding the observed DirectML lerp fallback.

CPU numerical equivalence is checked separately. GPU support and speed still
require a benchmark; do not change the ongoing matched-control protocol.
Only dense, real, first-order AdamW is supported.
"""
import math
import torch


class AdamWNoLerp(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        if lr < 0 or eps < 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW settings')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse or torch.is_complex(p):
                    raise ValueError('Only dense real gradients are supported')
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                step = state['step']
                avg, sq = state['exp_avg'], state['exp_avg_sq']
                p.mul_(1 - group['lr'] * group['weight_decay'])
                # Algebraically identical to lerp_(grad, 1-beta1), with different
                # floating-point rounding. These operations need DML validation.
                avg.mul_(beta1).add_(grad, alpha=1-beta1)
                sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = (sq.sqrt() / math.sqrt(1-beta2**step)).add_(group['eps'])
                p.addcdiv_(avg, denom, value=-group['lr']/(1-beta1**step))
        return loss

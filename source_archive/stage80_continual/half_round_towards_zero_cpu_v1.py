"""Independent CPU fp32-to-fp16 round-towards-zero numerical reference.

This contains no model inputs, lexical labels, lookup answers or device calls.
Use only to audit the saved DirectML cast; full generation parity stays required.
"""
import torch

def half_round_towards_zero(value):
    if value.device.type != 'cpu' or value.dtype != torch.float32:
        raise ValueError('CPU float32 values required')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Finite source required')
    nearest = value.half()
    if not bool(torch.isfinite(nearest).all()):
        raise ValueError('Overflowing half parameters refused')
    closer = torch.nextafter(nearest, torch.zeros_like(nearest))
    return torch.where(nearest.float().abs() > value.abs(), closer, nearest)

def self_check():
    # Positive/negative midpoints, representable values and small magnitudes.
    x = torch.tensor([1.0006, -1.0006, 1., -1., 0., 0.00000007, -0.00000007], dtype=torch.float32)
    y = half_round_towards_zero(x)
    assert bool((y.float().abs() <= x.abs()).all())
    assert torch.equal(y[:5], torch.tensor([1., -1., 1., -1., 0.], dtype=torch.float16))
    assert bool((torch.nextafter(y.abs(), torch.full_like(y, float('inf'))).float()[[0, 1, 5, 6]] > x.abs()[[0, 1, 5, 6]]).all())

if __name__ == '__main__':
    self_check()
    print('CPU RTZ reference checks PASS; no GPU/model call')

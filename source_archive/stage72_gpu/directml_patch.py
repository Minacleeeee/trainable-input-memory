"""
DirectML compatibility patch for torch-directml + transformers LLaMA models.

Root cause: transformers generates causal attention masks using torch.finfo(dtype).min
(-3.4e38 for float32). DirectML's internal uint8 conversion overflows on this value,
causing "RuntimeError: value cannot be converted to type uint8_t without overflow".

Fix: During training, do not pass attention_mask to the model (labels=-100 already
handles padding). For inference, patch the causal mask generator to use a safe value.
"""
import torch
from transformers.models.llama.modeling_llama import LlamaModel

SAFE_MASK_VALUE = -1e4


def patch_causal_mask():
    """Patch LlamaModel causal mask to use safe value (for inference with attention_mask)."""
    def patched_prepare(
        attention_mask, sequence_length, target_length, dtype, device,
        cache_position, batch_size, **kwargs
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=SAFE_MASK_VALUE,
            dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_sum = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(dtype)
            pm_float = (padding_sum == 0).to(dtype)
            causal_mask[:, :, :, :mask_length] = (
                causal_mask[:, :, :, :mask_length] * (1.0 - pm_float) + SAFE_MASK_VALUE * pm_float
            )
        return causal_mask

    LlamaModel._prepare_4d_causal_attention_mask_with_cache_position = staticmethod(patched_prepare)


def make_gpu_compatible(model):
    """Wrap model.forward to drop attention_mask during training (DirectML fix).

    labels=-100 already handles padding loss masking, so dropping attention_mask
    during training is mathematically equivalent for the loss computation.
    """
    original_forward = model.forward

    def gpu_forward(*args, **kwargs):
        if model.training and "attention_mask" in kwargs:
            kwargs.pop("attention_mask")
        return original_forward(*args, **kwargs)

    model.forward = gpu_forward
    return model


def get_directml_device():
    """Get the DirectML GPU device."""
    import torch_directml
    return torch_directml.device(0)
